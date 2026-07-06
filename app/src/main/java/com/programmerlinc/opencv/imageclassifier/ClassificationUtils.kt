//-----------------------------------------------------------------------
// <author>程序员Linc</author>
// <wechat>公众号：程序员Linc</wechat>
//-----------------------------------------------------------------------

package com.programmerlinc.opencv.imageclassifier

import kotlin.math.exp

internal data class Result(
    var detectedIndices: List<Int> = emptyList(),
    var detectedScore: MutableList<Float> = mutableListOf(),
    var processTimeMs: Long = 0
)

internal fun softMax(modelResult: FloatArray): FloatArray {
    val labelVals = modelResult.copyOf()
    val max = labelVals.max()
    var sum = 0.0f

    for (i in labelVals.indices) {
        labelVals[i] = exp(labelVals[i] - max)
        sum += labelVals[i]
    }

    if (sum != 0.0f) {
        for (i in labelVals.indices) {
            labelVals[i] /= sum
        }
    }

    return labelVals
}

internal fun getTop3(labelVals: FloatArray): List<Int> {
    val indices = mutableListOf<Int>()
    for (k in 0..2) {
        var max = 0.0f
        var idx = 0
        for (i in labelVals.indices) {
            val labelVal = labelVals[i]
            if (labelVal > max && !indices.contains(i)) {
                max = labelVal
                idx = i
            }
        }
        indices.add(idx)
    }
    return indices.toList()
}
