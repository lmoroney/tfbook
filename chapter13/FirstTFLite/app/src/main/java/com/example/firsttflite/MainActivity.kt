package com.example.firsttflite


import android.app.AlertDialog
import android.content.DialogInterface
import android.content.res.AssetManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.lang.Exception
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var tflite : Interpreter
    private lateinit var tflitemodel : ByteBuffer
    private lateinit var txtValue : EditText
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        try{
            tflitemodel = loadModelFile(this.assets, "model.tflite")
            tflite = Interpreter(tflitemodel)
        } catch(ex: Exception){
            ex.printStackTrace()
        }

        var convertButton: Button = findViewById<Button>(R.id.convertButton)
        convertButton.setOnClickListener{
            doInference()
        }

        txtValue = findViewById<EditText>(R.id.txtValue)
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun doInference(){
        var userVal: Float = txtValue.text.toString().toFloat()
        var inputVal: FloatArray = floatArrayOf(userVal)
        var outputVal: ByteBuffer = ByteBuffer.allocateDirect(4)
        outputVal.order(ByteOrder.nativeOrder())
        tflite.run(inputVal, outputVal)
        outputVal.rewind()
        var f:Float = outputVal.getFloat()
        val builder = AlertDialog.Builder(this)

        with(builder)
        {
            setTitle("TFLite Interpreter")
            setMessage("Your Value is:$f")
            setNeutralButton("OK", DialogInterface.OnClickListener {
                    dialog, id -> dialog.cancel()
            })
            show()
        }
    }
}
