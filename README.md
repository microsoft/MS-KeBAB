# Knowledge Base Construction and Access Benchmark (KeBAB)

This repository contains the backend implementation for the **K**nowledg**e** **B**ase construction and **A**ccess **B**enchmark (KeBAB🍢).

The recommended way to install the code is to clone the repository and install the package in editable mode:

```bash
pip install -e .[all]
```

# Structure

The repository is organized to ensure clarity and ease of navigation. Below is a brief overview of the main directories and their purposes:

* build/: Contains configuration files for automated builds.
* docs/: Documentation resources, including experiment results.
* kebab/: Contains the core implementation of the project, including all modules, utilities, and primary logic. 
    * configs/: Configuration files for the benchmark.
    * contracts/: Core interfaces and abstractions that define the project's key contracts and APIs for document, entity, task, etc.
    * tasks/: Task-specific implementations for various task types, such as extraction, linking, and more.
    * utils/: Utility functions for common operations across the project, including I/O handling, logging, and data processing.
    * mskebab.py: Contains the entry point class `Benchmark`.
* scripts/: Includes scripts for specific tasks such as data downloading, processing, or running experiments.
* tests/: Contains tests to ensure the robustness and reliability of the codebase.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
