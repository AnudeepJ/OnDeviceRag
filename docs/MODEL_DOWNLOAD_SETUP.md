# Model download setup

The app intentionally does not embed a Hugging Face token. EmbeddingGemma requires license
acceptance, and upstream exports can change over time. Production builds must therefore download
immutable, licensed copies from an HTTPS origin controlled by the product team.

## Publish the pinned artifacts

Publish these exact files under one versioned directory. Do not replace an object in place; use a
new directory and update the catalog when rolling a model.

| File | Exact bytes | SHA-256 |
| --- | ---: | --- |
| `gemma-4-E2B-it.litertlm` | 2,583,085,056 | `ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42` |
| `embeddinggemma-300m-seq512.tflite` | 179,132,472 | `ad09e81557203cb0e177abf9bf8727dfe138a7d394aa0f70f0b2ed16432e121a` |
| `embeddinggemma-tokenizer.model` | 4,689,074 | `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c` |

The origin must support HTTPS, stable `Content-Length`, HTTP range requests, and redirects that do
not require cookies or interactive login. Include the notices required by the applicable Gemma
license when distributing the model files.

## Configure a build

Add only the base URL to the developer's `~/.gradle/gradle.properties` or provide it to CI as the
Gradle project property `MODEL_CDN_BASE_URL`:

```properties
MODEL_CDN_BASE_URL=https://models.example.com/ondevice-rag/v1
```

The pinned SHA-256 values are non-secret integrity metadata and are already defaults in
`app/build.gradle.kts`. They can be overridden with `SHA256_GEMMA_E2B`, `SHA256_EMBEDDING`, and
`SHA256_TOKENIZER` only when deliberately publishing a new set of artifacts. E4B remains disabled
until its exact artifact and hash are configured.

Rebuild and install the app. The catalog Download buttons become enabled only when the URL is HTTPS
and the corresponding SHA-256 is valid.

## Device behavior

Downloads use Android `DownloadManager`, survive UI-process death, and are Wi-Fi-only by default
because the required set is about 2.8 GB. Completed artifacts are checked for exact byte length,
SHA-256 verified in a streaming pass, and atomically installed under app-private storage. Multiple
files may finish close together; verification/install operations are serialized in the inference
process. A failed install is surfaced in the model screen and can be retried.
