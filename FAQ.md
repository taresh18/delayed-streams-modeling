# FAQ

Here is the answer to a number of frequently asked questions.

### Torch compilation issues

With some PyTorch/triton versions, one might encounter compilation errors
like the following:
```
  Traceback (most recent call last):
  ...
  File "site-packages/torch/_inductor/runtime/triton_heuristics.py", line 1153, in make_launcher
    "launch_enter_hook": binary.__class__.launch_enter_hook,
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch._inductor.exc.InductorError: AttributeError: type object 'CompiledKernel' has no attribute 'launch_enter_hook'
```

If that's the case, you can disable torch compilation by setting the following
environment variable.
```bash
export NO_TORCH_COMPILE=1
```

### Issues installing the sentencepiece dependency

On some linux distributions (arch) or on macos, the local version of cmake can
be too recent for the sentencepiece dependency.

```
CMake Error at CMakeLists.txt:15 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

You can either downgrade your cmake version, e.g. 3.31.0 on arch works or try
setting `CMAKE_POLICY_VERSION_MINIMUM=3.5`.

If you run into some errors when compiling the sentencepiece rust bindings,
these could also be due to gcc being too recent, e.g. gcc 15. You can get
around this by using gcc-13, e.g. by setting the following after installing
the proper gcc packages.
```bash
export CMAKE_C_COMPILER=/usr/bin/gcc-13
export CMAKE_CXX_COMPILER=/usr/bin/g++-13 
CC=gcc-13 CXX=g++-13 cargo build --release
```

Alternatively you can set `CXXFLAGS="-include cstdint"`, see this
[issue](https://github.com/google/sentencepiece/issues/1108).

### Will you release training code?

Some finetuning code can be found in the [kyutai-labs/moshi-finetune repo](https://github.com/kyutai-labs/moshi-finetune).
This code has not been adapted to the Speech-To-Text and Text-To-Speech models
yet, but it should be a good starting point.


