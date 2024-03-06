# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2024-03-06

### Changed
- Separate description lines from summary lines in the docstrings.

## [0.6.1] - 2024-03-06

### Changed
- Update the docstrings to clarify the distinction between `split_dataset` and `split_dataset_v2`.

## [0.6.0] - 2024-03-06

### Added
- Add a new data splitting functions, `split_dataset_v2`.

## [0.5.0] - 2024-01-18

### Changed

- Update transformers version requirement to ^4.36.2.
- Move transformers and torch to a new extra dependency section named "transformers".

## [0.4.4] - 2023-12-22

### Changed

- Loosen pandas version requirement to ^1.5.3.

## [0.4.3] - 2023-11-02

### Changed

- Update mkdocs.yml to display documents for the new modules.

## [0.4.2] - 2023-11-02

### Added

- Add .env.template

### Changed

- Update test coverage config.

## [0.4.1] - 2023-11-02

### Changed

- Update documents for the recently added functions.

## [0.4.0] - 2023-11-02

### Added

- Add data splitting functions.
- Add evaluation metric report functions.

## [0.3.3] - 2023-10-30

### Added

- Add trim_tokens for text processing.

## [0.3.2] - 2023-10-27

### Changed

- Edit type hints
- Convert variables to appropriate types.

## [0.3.1] - 2023-10-27

### Added

- Add docstrings and type hints to the test scripts of the new utility modules.

## [0.3.0] - 2023-10-26

### Added

- Add utilities for training pipeline.
  - aws utils
  - git utils
  - io utils
  - wandb utils

## [0.2.0] - 2023-08-28

### Changed

- Change layout of module document.

## [0.1.1] - 2023-08-26

### Added

- Cover tests for more modules.

## [0.1.0] - 2023-08-26

### Added

- First release (pre-release) on PyPI.
