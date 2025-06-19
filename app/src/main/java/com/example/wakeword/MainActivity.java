package com.example.wakeword;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.ClipData;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CODE_PICK_WAV = 1234;
    private static final int PERMISSION_REQUEST_RECORD_AUDIO = 200;

    private TextView melspecText;
    private TextView wakeText;
    private TextView yamnetText;
    private Button btnPickWav;

    private ONNXModelRunner modelRunner;
    private Model model;
    private Interpreter yamnetInterpreter;
    private List<String> yamnetLabels;
    private AssetManager assetManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        melspecText = findViewById(R.id.melspec);
        wakeText    = findViewById(R.id.predicted);
        yamnetText  = findViewById(R.id.yamnet);
        btnPickWav  = findViewById(R.id.btn_pick_wav);
        assetManager = getAssets();

        try {
            modelRunner = new ONNXModelRunner(assetManager);
            model = new Model(modelRunner);
            loadYamNetModelAndLabels();
        } catch (Exception e) {
            Log.e("Initialization", "Error initializing models", e);
        }

        btnPickWav.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(
                        this,
                        new String[]{Manifest.permission.RECORD_AUDIO},
                        PERMISSION_REQUEST_RECORD_AUDIO
                );
                return;
            }
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("audio/*");
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
            startActivityForResult(intent, REQUEST_CODE_PICK_WAV);
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_RECORD_AUDIO
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            btnPickWav.performClick();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_PICK_WAV && resultCode == RESULT_OK && data != null) {
            runOnUiThread(() -> wakeText.setText(""));  // 초기화
            new Thread(() -> {
                StringBuilder resultBuilder = new StringBuilder();
                try {
                    if (data.getClipData() != null) {
                        ClipData clipData = data.getClipData();
                        for (int i = 0; i < clipData.getItemCount(); i++) {
                            Uri uri = clipData.getItemAt(i).getUri();
                            String result = processOneFile(uri, "wav" + (i + 1));
                            resultBuilder.append(result).append("\n");
                        }
                    } else if (data.getData() != null) {
                        Uri uri = data.getData();
                        String result = processOneFile(uri, "wav1");
                        resultBuilder.append(result).append("\n");
                    }
                } catch (Exception e) {
                    Log.e("WAV", "Error reading WAV files", e);
                }
                runOnUiThread(() -> wakeText.setText(resultBuilder.toString()));
            }).start();
        }
    }

    private String processOneFile(Uri uri, String label) {
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            short[] pcm16 = readWavToPCM16(is);
            final int FRAMES = 1280;
            float[] floatBuf = new float[FRAMES];
            for (int offset = 0; offset + FRAMES <= pcm16.length; offset += FRAMES) {
                for (int i = 0; i < FRAMES; i++) {
                    floatBuf[i] = pcm16[offset + i] / 32768f;
                }
                String probStr = model.predict_WakeWord(floatBuf);
                float prob = Float.parseFloat(probStr);
                if (prob > 0.85f) {
                    return label + ": Detected!";
                }
            }
            return label + ": x";
        } catch (Exception e) {
            Log.e("WAV", "Error processing " + label, e);
            return label + ": Error";
        }
    }

    private short[] readWavToPCM16(InputStream is) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(is));
        byte[] header = new byte[44];
        dis.readFully(header);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int len;
        while ((len = dis.read(buf)) > 0) {
            baos.write(buf, 0, len);
        }
        byte[] pcmBytes = baos.toByteArray();
        short[] pcm16 = new short[pcmBytes.length / 2];
        ByteBuffer bb = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < pcm16.length; i++) {
            pcm16[i] = bb.getShort();
        }
        return pcm16;
    }

    private void loadYamNetModelAndLabels() throws IOException {
        AssetFileDescriptor fd = assetManager.openFd("yamnet.tflite");
        try (FileInputStream fis = new FileInputStream(fd.getFileDescriptor())) {
            FileChannel fc = fis.getChannel();
            MappedByteBuffer modelBuffer = fc.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.getStartOffset(),
                    fd.getDeclaredLength()
            );
            yamnetInterpreter = new Interpreter(modelBuffer);
        }
        yamnetLabels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(assetManager.open("yamnet_labels.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                yamnetLabels.add(line);
            }
        }
    }
}