//-----------------------------------------------------------------------
// <author>程序员Linc</author>
// <wechat>公众号：程序员Linc</wechat>
//-----------------------------------------------------------------------

package com.programmerlinc.opencv.imageclassifier

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc

internal object OpenCVClassifier {
    private val IMAGENET_MEAN = Scalar(0.485, 0.456, 0.406)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun createBlob(rgbFrame: Mat): Mat {
        val blob = Dnn.blobFromImage(
            rgbFrame,
            1.0 / 255.0,
            Size(IMAGE_SIZE_X.toDouble(), IMAGE_SIZE_Y.toDouble()),
            IMAGENET_MEAN,
            false,
            false
        )
        divideBlobByStd(blob)
        return blob
    }

    private fun divideBlobByStd(blob: Mat) {
        val planeSize = IMAGE_SIZE_X * IMAGE_SIZE_Y
        val data = FloatArray(3 * planeSize)
        blob.get(0, 0, data)
        for (channel in 0 until 3) {
            val offset = channel * planeSize
            val std = IMAGENET_STD[channel]
            for (i in 0 until planeSize) {
                data[offset + i] /= std
            }
        }
        blob.put(0, 0, data)
    }

    fun matToFloatArray(output: Mat): FloatArray {
        val result = FloatArray(output.total().toInt() * output.channels())
        output.get(0, 0, result)
        return result
    }

    fun classify(net: Net, rgbFrame: Mat): Result {
        var blob: Mat? = null
        var output: Mat? = null
        return try {
            val startTime = System.currentTimeMillis()
            blob = createBlob(rgbFrame)
            net.setInput(blob)
            output = net.forward()
            val processTimeMs = System.currentTimeMillis() - startTime

            val probabilities = softMax(matToFloatArray(output))
            val topIndices = getTop3(probabilities)
            Result(
                detectedIndices = topIndices,
                detectedScore = topIndices.map { probabilities[it] }.toMutableList(),
                processTimeMs = processTimeMs
            )
        } finally {
            blob?.release()
            output?.release()
        }
    }

    fun drawResults(frame: Mat, labels: List<String>, result: Result) {
        val baseY = 30.0
        val lineHeight = 28.0
        result.detectedIndices.forEachIndexed { index, classId ->
            val score = result.detectedScore[index]
            val label = "${labels[classId]}: %.1f%%".format(score * 100)
            val y = baseY + index * lineHeight
            val baseline = IntArray(1)
            val labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, 2, baseline)
            Imgproc.rectangle(
                frame,
                Point(8.0, y - labelSize.height - 4),
                Point(16.0 + labelSize.width, y + baseline[0].toDouble() + 4),
                Scalar(255.0, 255.0, 255.0),
                Imgproc.FILLED
            )
            Imgproc.putText(
                frame,
                label,
                Point(12.0, y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0.0, 128.0, 0.0),
                2
            )
        }
        Imgproc.putText(
            frame,
            "${result.processTimeMs}ms",
            Point(12.0, frame.rows() - 16.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(255.0, 255.0, 255.0),
            1
        )
    }
}
