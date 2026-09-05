package com.example.pdfgemmarag.core.process

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.os.Build
import android.os.Process

/**
 * `Application.onCreate` runs once per process. The app has two: the default `:ui` process and
 * `:inference`. Heavy native runtimes must never be touched from `:ui`, and Compose must never be
 * inflated from `:inference`.
 */
object ProcessGuard {
    const val INFERENCE_SUFFIX = ":inference"

    fun currentProcessName(context: Context): String {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            return Application.getProcessName()
        }
        val pid = Process.myPid()
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return am.runningAppProcesses?.firstOrNull { it.pid == pid }?.processName ?: context.packageName
    }

    fun isInferenceProcess(context: Context): Boolean =
        currentProcessName(context).endsWith(INFERENCE_SUFFIX)

    fun isUiProcess(context: Context): Boolean = !isInferenceProcess(context)

    /** Throws if called from the wrong process; used as a cheap assertion in constructors. */
    fun requireInferenceProcess(context: Context) {
        check(isInferenceProcess(context)) {
            "This component must only be created in the $INFERENCE_SUFFIX process " +
                "(current: ${currentProcessName(context)})"
        }
    }
}
