//-----------------------------------------------------------------------
// <author>程序员Linc</author>
// <wechat>公众号：程序员Linc</wechat>
//-----------------------------------------------------------------------

package com.programmerlinc.opencv.imageclassifier

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.programmerlinc.opencv.imageclassifier", appContext.packageName)
    }

    @Test
    fun loadModelWithOpenCV() {
        assert(OpenCVLoader.initLocal())
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val net = OpenCVModelLoader.loadFromRawResource(
            appContext.resources,
            R.raw.mobilenetv2_fp32
        )
        assertNotNull(net)
    }
}
