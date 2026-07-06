//-----------------------------------------------------------------------
// <author>程序员Linc</author>
// <wechat>公众号：程序员Linc</wechat>
//-----------------------------------------------------------------------

package com.programmerlinc.opencv.imageclassifier

import com.programmerlinc.opencv.imageclassifier.databinding.ActivityMainBinding
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import org.opencv.android.CameraActivity
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.util.Collections
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : CameraActivity(), CvCameraViewListener2 {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraView: CameraBridgeViewBase

    private val labelData: List<String> by lazy { readLabels() }
    private val isProcessing = AtomicBoolean(false)

    @Volatile
    private var net: Net? = null
    private var enableQuantizedModel = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed!")
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        Log.i(TAG, "OpenCV loaded successfully")

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraView = binding.cameraView
        cameraView.visibility = CameraBridgeViewBase.VISIBLE
        cameraView.setCvCameraViewListener(this)

        reloadModel()
    }

    override fun onResume() {
        super.onResume()
        cameraView.enableView()
    }

    override fun onPause() {
        super.onPause()
        cameraView.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraView.disableView()
        net = null
    }

    override fun getCameraViewList(): List<CameraBridgeViewBase> {
        return Collections.singletonList(cameraView)
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.d(TAG, "Camera started: ${width}x$height")
    }

    override fun onCameraViewStopped() {
        Log.d(TAG, "Camera stopped")
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val frame = inputFrame.rgba()
        val currentNet = net ?: return frame

        if (!isProcessing.compareAndSet(false, true)) {
            return frame
        }

        var rgb: Mat? = null
        try {
            rgb = Mat()
            Imgproc.cvtColor(frame, rgb, Imgproc.COLOR_RGBA2RGB)

            val result = OpenCVClassifier.classify(currentNet, rgb)
            OpenCVClassifier.drawResults(frame, labelData, result)
            runOnUiThread { updateUI(result) }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
        } finally {
            rgb?.release()
            isProcessing.set(false)
        }

        return frame
    }

    private fun reloadModel() {
        val modelResId = if (enableQuantizedModel) {
            R.raw.mobilenetv2_int8
        } else {
            R.raw.mobilenetv2_fp32
        }

        net = OpenCVModelLoader.loadFromRawResource(resources, modelResId)
        if (net == null) {
            Toast.makeText(
                this,
                "Failed to load MobileNet model for OpenCV DNN.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun updateUI(result: Result) {
        if (result.detectedScore.isEmpty()) {
            return
        }

        binding.percentMeter.progress = (result.detectedScore[0] * 100).toInt()
        binding.detectedItem1.text = labelData[result.detectedIndices[0]]
        binding.detectedItemValue1.text = "%.2f%%".format(result.detectedScore[0] * 100)

        if (result.detectedIndices.size > 1) {
            binding.detectedItem2.text = labelData[result.detectedIndices[1]]
            binding.detectedItemValue2.text = "%.2f%%".format(result.detectedScore[1] * 100)
        }

        if (result.detectedIndices.size > 2) {
            binding.detectedItem3.text = labelData[result.detectedIndices[2]]
            binding.detectedItemValue3.text = "%.2f%%".format(result.detectedScore[2] * 100)
        }

        binding.inferenceTimeValue.text = "${result.processTimeMs}ms"
    }

    private fun readLabels(): List<String> {
        return resources.openRawResource(R.raw.imagenet_classes).bufferedReader().readLines()
    }

    companion object {
        private const val TAG = "OpenCVImageClassifier"
    }
}
