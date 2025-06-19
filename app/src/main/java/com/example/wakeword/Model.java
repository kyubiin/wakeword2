package com.example.wakeword;

import android.content.res.AssetManager;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

// ONNX 모델을 로드하고 추론을 수행하는 클래스
class ONNXModelRunner {

    private static final int BATCH_SIZE = 1; // 배치 크기 (한 번에 처리할 샘플 수)

    AssetManager assetManager;      // assets 폴더 접근 용 매니저
    OrtSession wakewordSession;     // 웨이크워드 모델 세션
    OrtEnvironment environment = OrtEnvironment.getEnvironment(); // ONNX 런타임 환경

    // 생성자: AssetManager를 받아 모델 파일을 로드하여 세션 생성
    public ONNXModelRunner(AssetManager assetManager) throws IOException, OrtException {
        this.assetManager = assetManager;
        try {
            // chaamiiya.onnx 파일을 읽어 세션 초기화
            wakewordSession = environment.createSession(readModelFile(assetManager, "chaamiiya.onnx"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 입력 오디오 배열로부터 Mel 스펙트로그램을 생성하는 메서드
     * 내부적으로 melspectrogram.onnx 모델을 로드하고 실행
     */
    public float[][] get_mel_spectrogram(float[] inputArray) throws OrtException, IOException {
        OrtSession session;
        try (InputStream modelInputStream = assetManager.open("melspectrogram.onnx")) {
            // 모델 파일 바이트로 읽기
            byte[] modelBytes = new byte[modelInputStream.available()];
            modelInputStream.read(modelBytes);
            session = OrtEnvironment.getEnvironment().createSession(modelBytes);
        }
        float[][] outputArray = null;
        int SAMPLES = inputArray.length;
        // FloatBuffer로 tensor 생성
        FloatBuffer floatBuffer = FloatBuffer.wrap(inputArray);
        OnnxTensor inputTensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                floatBuffer,
                new long[]{BATCH_SIZE, SAMPLES}
        );

        try (OrtSession.Result results = session.run(
                Collections.singletonMap(
                        session.getInputNames().iterator().next(),
                        inputTensor
                )
        )) {
            // 결과는 4차원 텐서로 불러옴
            float[][][][] outputTensor = (float[][][][]) results.get(0).getValue();
            // 불필요 차원 제거
            float[][] squeezed = squeeze(outputTensor);
            // Mel 스펙트로그램 변환 적용
            outputArray = applyMelSpecTransform(squeezed);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (inputTensor != null) inputTensor.close();
            if (session != null) session.close();
        }
        // 환경 닫기 (세션별로 열면 매번 닫아야 함)
        OrtEnvironment.getEnvironment().close();
        return outputArray;
    }

    /**
     * 4차원 배열에서 [0][0] 차원 제거 (squeeze)
     */
    public static float[][] squeeze(float[][][][] originalArray) {
        int rows = originalArray[0][0].length;
        int cols = originalArray[0][0][0].length;
        float[][] squeezedArray = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                squeezedArray[i][j] = originalArray[0][0][i][j];
            }
        }
        return squeezedArray;
    }

    /**
     * Mel 스펙 변환 예시: 값 범위 조정 등의 후처리
     */
    public static float[][] applyMelSpecTransform(float[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        float[][] transformedArray = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // 예시: 10으로 나눈 후 2 더하기
                transformedArray[i][j] = array[i][j] / 10.0f + 2.0f;
            }
        }
        return transformedArray;
    }

    /**
     * 임베딩 모델을 실행하여 embedding 벡터 생성
     */
    public float[][] generateEmbeddings(float[][][][] input) throws OrtException, IOException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        InputStream is = assetManager.open("embedding_model.onnx");
        byte[] model = new byte[is.available()];
        is.read(model);
        is.close();

        OrtSession sess = env.createSession(model);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, input);
        try (OrtSession.Result results = sess.run(
                Collections.singletonMap("input_1", inputTensor)
        )) {
            float[][][][] rawOutput = (float[][][][]) results.get(0).getValue();
            // 출력 형태 변경: [batch][features]
            float[][] reshapedOutput = new float[rawOutput.length][rawOutput[0][0][0].length];
            for (int i = 0; i < rawOutput.length; i++) {
                System.arraycopy(rawOutput[i][0][0], 0, reshapedOutput[i], 0, rawOutput[i][0][0].length);
            }
            return reshapedOutput;
        } catch (Exception e) {
            Log.d("exception", "not_predicted " + e.getMessage());
        } finally {
            if (inputTensor != null) inputTensor.close();
            if (sess != null) sess.close();
        }
        env.close();
        return null;
    }

    /**
     * wakeword 모델로부터 확률 예측
     */
    public String predictWakeWord(float[][][] inputArray) throws OrtException {
        float[][] result;
        String resultant = "";
        OnnxTensor inputTensor = null;
        try {
            inputTensor = OnnxTensor.createTensor(environment, inputArray);
            OrtSession.Result outputs = wakewordSession.run(
                    Collections.singletonMap(
                            wakewordSession.getInputNames().iterator().next(),
                            inputTensor
                    )
            );
            result = (float[][]) outputs.get(0).getValue();
            // 확률을 문자열로 포맷
            resultant = String.format("%.5f", (double) result[0][0]);
        } catch (OrtException e) {
            e.printStackTrace();
        } finally {
            if (inputTensor != null) inputTensor.close();
        }
        return resultant;
    }

    // AssetManager를 이용해 모델 filebyte로 읽어옴
    private byte[] readModelFile(AssetManager assetManager, String filename) throws IOException {
        try (InputStream is = assetManager.open(filename)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            return buffer;
        }
    }
}


/**
 * input 오디오 데이터를 버퍼링하고, 프레임 단위로 Mel 스펙 및 임베딩을 계산하는 main 모델 클래스
 */
public class Model {
    int n_prepared_samples = 1280;            // 준비된 샘플 수
    int sampleRate = 16000;                   // 샘플링 레이트
    int melspectrogramMaxLen = 10 * 97;       // 최대 Mel 스펙 길이
    int feature_buffer_max_len = 120;         // 특징 버퍼 최대 길이
    ONNXModelRunner modelRunner;              // ONNX 실행기
    float[][] featureBuffer;                  // 임베딩 결과 버퍼
    ArrayDeque<Float> raw_data_buffer = new ArrayDeque<>(sampleRate * 10); // 원시 오디오 버퍼
    float[] raw_data_remainder = new float[0]; // 남은 샘플
    float[][] melspectrogramBuffer;           // Mel 스펙 버퍼
    int accumulated_samples = 0;              // 누적 샘플

    // 생성자: 기본 Mel 스펙 버퍼 초기화 및 더미 임베딩 생성
    Model(ONNXModelRunner modelRunner) {
        melspectrogramBuffer = new float[76][32];
        for (int i = 0; i < 76; i++) {
            for (int j = 0; j < 32; j++) {
                melspectrogramBuffer[i][j] = 1.0f; // numpy.ones 시뮬레이션
            }
        }
        this.modelRunner = modelRunner;
        try {
            // 랜덤 데이터로 초기 임베딩 생성
            this.featureBuffer = this._getEmbeddings(
                    this.generateRandomIntArray(16000 * 4),
                    76,
                    8
            );
        } catch (Exception e) {
            System.out.print(e.getMessage());
        }
    }

    /**
     * 특징 행렬을 얻기 위한 프레임 버퍼링 및 슬라이딩 윈도우
     */
    public float[][][] getFeatures(int nFeatureFrames, int startNdx) {
        int endNdx;
        if (startNdx != -1) {
            endNdx = (startNdx + nFeatureFrames != 0)
                    ? (startNdx + nFeatureFrames)
                    : featureBuffer.length;
        } else {
            startNdx = Math.max(0, featureBuffer.length - nFeatureFrames);
            endNdx = featureBuffer.length;
        }
        int length = endNdx - startNdx;
        float[][][] result = new float[1][length][featureBuffer[0].length];
        for (int i = 0; i < length; i++) {
            System.arraycopy(
                    featureBuffer[startNdx + i],
                    0,
                    result[0][i],
                    0,
                    featureBuffer[startNdx + i].length
            );
        }
        return result;
    }

    /**
     * Mel 스펙트로그램 → 임베딩 변환 파이프라인
     */
    private float[][] _getEmbeddings(float[] x, int windowSize, int stepSize)
            throws OrtException, IOException {
        // Mel 스펙트로그램 계산
        float[][] spec = this.modelRunner.get_mel_spectrogram(x);
        // 슬라이딩 윈도우로 프레임 분할
        ArrayList<float[][]> windows = new ArrayList<>();
        for (int i = 0; i <= spec.length - windowSize; i += stepSize) {
            float[][] window = new float[windowSize][spec[0].length];
            for (int j = 0; j < windowSize; j++) {
                System.arraycopy(spec[i + j], 0, window[j], 0, spec[0].length);
            }
            windows.add(window);
        }
        // 배치 형태로 변환 (4차원 텐서)
        float[][][][] batch = new float[windows.size()][windowSize][spec[0].length][1];
        for (int i = 0; i < windows.size(); i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int k = 0; k < spec[0].length; k++) {
                    batch[i][j][k][0] = windows.get(i)[j][k];
                }
            }
        }
        // 임베딩 생성
        return modelRunner.generateEmbeddings(batch);
    }

    /**
     * 랜덤 정수 배열 생성 (데이터 시뮬레이션 용)
     */
    private float[] generateRandomIntArray(int size) {
        float[] arr = new float[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            arr[i] = (float) random.nextInt(2000) - 1000; // [-1000,1000)
        }
        return arr;
    }

    /**
     * input 오디오 데이터를 internal buffer에 추가
     */
    public void bufferRawData(float[] x) {
        if (x != null) {
            while (raw_data_buffer.size() + x.length > sampleRate * 10) {
                raw_data_buffer.poll();
            }
            for (float value : x) {
                raw_data_buffer.offer(value);
            }
        }
    }

    /**
     * 스트리밍 Mel 스펙트로그램 계산 후 버퍼 관리
     */
    public void streamingMelSpectrogram(int n_samples) {
        if (raw_data_buffer.size() < 400) {
            throw new IllegalArgumentException("최소 400 샘플 필요");
        }
        float[] tempArray = new float[n_samples + 480];
        Object[] rawDataArray = raw_data_buffer.toArray();
        int startIdx = Math.max(0, rawDataArray.length - n_samples - 480);
        for (int i = startIdx; i < rawDataArray.length; i++) {
            tempArray[i - startIdx] = (Float) rawDataArray[i];
        }
        float[][] new_mel;
        try {
            new_mel = modelRunner.get_mel_spectrogram(tempArray);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        // 기존 버퍼와 합치고 최대 길이 유지
        int oldLen = melspectrogramBuffer.length;
        int newLen = new_mel.length;
        float[][] combined = new float[oldLen + newLen][];
        System.arraycopy(melspectrogramBuffer, 0, combined, 0, oldLen);
        System.arraycopy(new_mel, 0, combined, oldLen, newLen);
        melspectrogramBuffer = combined;
        if (melspectrogramBuffer.length > melspectrogramMaxLen) {
            float[][] trimmed = new float[melspectrogramMaxLen][];
            int offset = melspectrogramBuffer.length - melspectrogramMaxLen;
            System.arraycopy(melspectrogramBuffer, offset, trimmed, 0, melspectrogramMaxLen);
            melspectrogramBuffer = trimmed;
        }
    }

    /**
     * 스트리밍 feature 추출: 오디오 청크 단위로 Embedding 생성 및 버퍼 업데이트
     */
    public int streaming_features(float[] audiobuffer) {
        int processed_samples = 0;
        this.accumulated_samples = 0;
        // 남은 샘플과 합쳐서 처리
        if (raw_data_remainder.length != 0) {
            int rem = raw_data_remainder.length;
            float[] concat = new float[rem + audiobuffer.length];
            System.arraycopy(raw_data_remainder, 0, concat, 0, rem);
            System.arraycopy(audiobuffer, 0, concat, rem, audiobuffer.length);
            audiobuffer = concat;
            raw_data_remainder = new float[0];
        }
        // 1280 샘플 단위로 bufferRawData 호출 및 remainder 관리
        int total = this.accumulated_samples + audiobuffer.length;
        int remainder = total % 1280;
        if (total >= 1280) {
            float[] evenChunks = audiobuffer;
            if (remainder != 0) {
                evenChunks = new float[audiobuffer.length - remainder];
                System.arraycopy(audiobuffer, 0, evenChunks, 0, evenChunks.length);
                raw_data_remainder = new float[remainder];
                System.arraycopy(audiobuffer, evenChunks.length, raw_data_remainder, 0, remainder);
            }
            this.bufferRawData(evenChunks);
            this.accumulated_samples += evenChunks.length;
        } else {
            this.accumulated_samples += audiobuffer.length;
            this.bufferRawData(audiobuffer);
        }
        // 충분히 모이면 Mel+Embedding
        if (this.accumulated_samples >= 1280 && this.accumulated_samples % 1280 == 0) {
            this.streamingMelSpectrogram(this.accumulated_samples);
            float[][][][] x = new float[1][76][32][1];
            int chunks = this.accumulated_samples / 1280;
            for (int i = chunks - 1; i >= 0; i--) {
                int ndx = -8 * i;
                if (ndx == 0) ndx = melspectrogramBuffer.length;
                int start = Math.max(0, ndx - 76);
                int end = ndx;
                for (int j = start, k = 0; j < end; j++, k++) {
                    for (int w = 0; w < 32; w++) {
                        x[0][k][w][0] = melspectrogramBuffer[j][w];
                    }
                }
                if (x[0].length == 76) {
                    try {
                        float[][] newFeatures = modelRunner.generateEmbeddings(x);
                        // featureBuffer 확장
                        if (featureBuffer == null) {
                            featureBuffer = newFeatures;
                        } else {
                            int oldFB = featureBuffer.length;
                            int newFB = newFeatures.length;
                            int cols = featureBuffer[0].length;
                            float[][] updated = new float[oldFB + newFB][cols];
                            for (int l = 0; l < oldFB; l++) {
                                System.arraycopy(featureBuffer[l], 0, updated[l], 0, cols);
                            }
                            for (int k = 0; k < newFB; k++) {
                                System.arraycopy(newFeatures[k], 0, updated[oldFB + k], 0, cols);
                            }
                            featureBuffer = updated;
                        }
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            processed_samples = this.accumulated_samples;
            this.accumulated_samples = 0;
        }
        // featureBuffer 크기 제한
        if (featureBuffer.length > feature_buffer_max_len) {
            int max = feature_buffer_max_len;
            int cols = featureBuffer[0].length;
            float[][] trimmed = new float[max][cols];
            int offset = featureBuffer.length - max;
            for (int i = 0; i < max; i++) {
                trimmed[i] = featureBuffer[offset + i];
            }
            featureBuffer = trimmed;
        }
        return processed_samples != 0 ? processed_samples : this.accumulated_samples;
    }

    /**
     * 오디오 버퍼를 받아 최종 wakeword 확률 반환
     */
    public String predict_WakeWord(float[] audiobuffer) {
        n_prepared_samples = this.streaming_features(audiobuffer);
        float[][][] res = this.getFeatures(16, -1);
        String result = "";
        try {
            result = modelRunner.predictWakeWord(res);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return result;
    }
}

