//-----------------------------------------------------------------------
// <author>程序员Linc</author>
// <wechat>公众号：程序员Linc</wechat>
//-----------------------------------------------------------------------

package com.programmerlinc.opencv.imageclassifier

import android.content.res.Resources
import android.util.Log
import org.opencv.core.MatOfByte
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import java.io.IOException

internal object OpenCVModelLoader {
    private const val TAG = "OpenCVModelLoader"

    fun loadFromRawResource(resources: Resources, modelResId: Int): Net? {
        if (modelResId == 0) {
            Log.e(TAG, "Model resource id is 0")
            return null
        }

        val modelBuffer = loadFileFromResource(resources, modelResId) ?: return null
        return try {
            val net = Dnn.readNetFromONNX(modelBuffer)
            Log.i(TAG, "OpenCV DNN network loaded successfully")
            net
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model with OpenCV DNN", e)
            null
        } finally {
            modelBuffer.release()
        }
    }

    private fun loadFileFromResource(resources: Resources, id: Int): MatOfByte? {
        return try {
            resources.openRawResource(id).use { inputStream ->
                val buffer = inputStream.readBytes()
                MatOfByte(*buffer)
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read model from resources", e)
            null
        }
    }
}
