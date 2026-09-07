plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.kotlin.serialization)
  alias(libs.plugins.kotlin.parcelize)
  alias(libs.plugins.ksp)
}

// Apryse license key: put PDFTRON_LICENSE_KEY=... in ~/.gradle/gradle.properties (never commit it).
val pdftronLicenseKey: String = (project.findProperty("PDFTRON_LICENSE_KEY") as String?) ?: ""

android {
  namespace = "com.example.pdfgemmarag"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.example.pdfgemmarag"
    minSdk = 29
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"

    // Gemma on GPU is 64-bit only; shipping arm64-v8a alone halves APK size.
    ndk { abiFilters += listOf("arm64-v8a") }

    manifestPlaceholders["pdftronLicenseKey"] = pdftronLicenseKey
    multiDexEnabled = true
    vectorDrawables.useSupportLibrary = true
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

    // Model hosting. HF-gated weights must be mirrored on your own CDN; SHA-256 values gate install.
    fun prop(name: String, default: String) = "\"${(project.findProperty(name) as String?) ?: default}\""
    buildConfigField("String", "MODEL_CDN_BASE_URL", prop("MODEL_CDN_BASE_URL", "https://models.example.invalid/ondevice-rag"))
    // Hashes are integrity metadata, not secrets. Keeping the hashes for the pinned artifacts in
    // source means a developer only has to configure the CDN location; CI may still override them
    // when intentionally rolling a model artifact.
    buildConfigField("String", "SHA256_GEMMA_E2B", prop("SHA256_GEMMA_E2B", "ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42"))
    buildConfigField("String", "SHA256_GEMMA_E4B", prop("SHA256_GEMMA_E4B", ""))
    buildConfigField("String", "SHA256_EMBEDDING", prop("SHA256_EMBEDDING", "ad09e81557203cb0e177abf9bf8727dfe138a7d394aa0f70f0b2ed16432e121a"))
    buildConfigField("String", "SHA256_TOKENIZER", prop("SHA256_TOKENIZER", "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"))
  }

  buildTypes {
    debug {
      // x86_64 lets the CPU path run on the emulator; GPU/NPU need real hardware.
      ndk { abiFilters += listOf("x86_64") }
    }
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  sourceSets {
    // Tokenizer parity fixtures are shared between the JVM parity test and the on-device SelfTest.
    getByName("main").assets.srcDir("src/test/resources")
  }

  buildFeatures {
    compose = true
    aidl = true
    buildConfig = true
    shaders = false
  }

  packaging {
    jniLibs {
      // litertlm-android and litert both bundle libLiteRt.so; keep one copy.
      pickFirsts += "**/libLiteRt.so"
      useLegacyPackaging = false
    }
    resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
  }
}

kotlin { jvmToolchain(17) }

ksp { arg("room.schemaLocation", "$projectDir/schemas") }

dependencies {
  val composeBom = platform(libs.androidx.compose.bom)
  implementation(composeBom)
  androidTestImplementation(composeBom)

  // Core
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.service)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(libs.androidx.documentfile)
  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.kotlinx.coroutines.guava)
  implementation(libs.kotlinx.coroutines.play.services)

  // Compose
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.material.icons.extended)
  debugImplementation(libs.androidx.compose.ui.tooling)
  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
  debugImplementation(libs.androidx.compose.ui.test.manifest)

  // Navigation 3
  implementation(libs.androidx.navigation3.ui)
  implementation(libs.androidx.navigation3.runtime)
  implementation(libs.androidx.lifecycle.viewmodel.navigation3)

  // AppSearch 1.1.0 (vector store). The @Document class is written in Java so the
  // annotation processor runs under javac (kapt is incompatible with AGP built-in Kotlin).
  implementation(libs.androidx.appsearch)
  implementation(libs.androidx.appsearch.local.storage)
  annotationProcessor(libs.androidx.appsearch.compiler)
  implementation(libs.androidx.concurrent.futures.ktx)

  // LiteRT-LM (Gemma 4) and LiteRT (EmbeddingGemma)
  implementation(libs.litertlm.android)
  implementation(libs.litert)

  // Room (chat transcripts, documents) via KSP
  implementation(libs.androidx.room.runtime)
  implementation(libs.androidx.room.ktx)
  ksp(libs.androidx.room.compiler)

  // PDF + OCR
  implementation(libs.pdftron)
  implementation(libs.mlkit.text.recognition)
  implementation(libs.mlkit.text.recognition.chinese)
  implementation(libs.mlkit.text.recognition.japanese)
  implementation(libs.mlkit.text.recognition.korean)

  // Tests
  testImplementation(libs.junit)
  testImplementation(libs.org.json) // real org.json for JSON fixtures in JVM tests
  testImplementation(libs.kotlinx.coroutines.test)
  androidTestImplementation(libs.androidx.test.core)
  androidTestImplementation(libs.androidx.test.ext.junit)
  androidTestImplementation(libs.androidx.test.runner)
  androidTestImplementation(libs.androidx.test.espresso.core)
}
