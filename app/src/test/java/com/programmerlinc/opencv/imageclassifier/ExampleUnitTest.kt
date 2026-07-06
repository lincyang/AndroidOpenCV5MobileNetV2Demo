//-----------------------------------------------------------------------
// <author>程序员Linc</author>
// <wechat>公众号：程序员Linc</wechat>
//-----------------------------------------------------------------------

package com.programmerlinc.opencv.imageclassifier

import org.junit.Assert.assertEquals
import org.junit.Test

class ExampleUnitTest {
    @Test
    fun softMaxSumsToOne() {
        val result = softMax(floatArrayOf(1.0f, 2.0f, 3.0f))
        val sum = result.sum()
        assertEquals(1.0f, sum, 0.001f)
    }
}
