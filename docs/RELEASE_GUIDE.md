# GitHub Release Creation Guide

## Quick Start

### Prerequisites
1. Install GitHub CLI: https://cli.github.com/
2. Authenticate with GitHub: `gh auth login`
3. Ensure you're in the Cellex project directory

### Create Release v2.3.0
```powershell
.\create-release.ps1 -Version "2.3.0"
```

### Advanced Options
```powershell
# Create draft release
.\create-release.ps1 -Version "2.3.0" -Draft

# Create prerelease
.\create-release.ps1 -Version "2.3.0" -Prerelease

# Specify custom model path
.\create-release.ps1 -Version "2.3.0" -ModelPath "models/best_model_epoch_2.pth"

# Combined options
.\create-release.ps1 -Version "2.3.0" -ModelPath "models/best_model_epoch_2.pth" -Draft
```

## What's Included in Each Release

### Files Packaged:
- **Core Files**: README.md, requirements.txt, setup.py
- **Scripts**: predict_image.py, train.py, verify_dataset.py, download_datasets.ps1  
- **Source Code**: Complete `src/` directory with all modules
- **Configuration**: `config/` directory with training settings
- **Documentation**: `docs/` directory with guides and changelog
- **Tests**: `tests/` directory for validation

### Model Files:
- Automatically detects and includes the best available model
- Renames model to `cellex-v{VERSION}.pth` format
- Generates `MODEL_INFO.md` with performance metrics

### Generated Files:
- `cellex-v{VERSION}.zip` - Complete release package
- `MODEL_INFO.md` - Model specifications and performance data
- Release notes extracted from `CHANGELOG.md`

## Release Process

1. **Validation**: Checks GitHub CLI installation and authentication
2. **Model Detection**: Finds best available model file  
3. **Package Creation**: Assembles all release files
4. **Release Upload**: Creates GitHub release with assets
5. **Cleanup**: Optional removal of temporary files

## Troubleshooting

### Common Issues:
- **GitHub CLI not found**: Install from https://cli.github.com/
- **Not authenticated**: Run `gh auth login`
- **Invalid version format**: Use semantic versioning (e.g., 2.3.0)
- **No model found**: Ensure models exist in `models/` directory

### Version Naming:
- Use semantic versioning: MAJOR.MINOR.PATCH
- Examples: 2.3.0, 2.3.1, 3.0.0
- Model files will be renamed to: `cellex-v{VERSION}.pth`

## Example Output

```
🚀 CELLEX GITHUB RELEASE CREATOR v1.0
==================================================
Creating release for version: 2.3.0

🔍 Checking prerequisites...
✅ GitHub CLI found: gh version 2.40.1
✅ Authenticated with GitHub
✅ All prerequisites met!

🔍 Finding best model...
✅ Found model: best_model_epoch_2.pth

📝 Extracting release notes from CHANGELOG...
✅ Extracted release notes from CHANGELOG

📦 Creating release package...
  ✅ Added: README.md
  ✅ Added: requirements.txt
  ✅ Added directory: src
  ✅ Added directory: config
  ✅ Added versioned model: cellex-v2.3.0.pth
✅ Created release package: cellex-v2.3.0.zip

🚀 Creating GitHub release...
🎉 Successfully created release v2.3.0!
🔗 Release URL: https://github.com/juliuspleunes4/cellex/releases/tag/v2.3.0

🎉 RELEASE CREATION COMPLETED SUCCESSFULLY!
```

## Security Notes
- Script requires GitHub authentication
- Only includes necessary project files (no sensitive data)
- Temporary files are cleaned up after execution
- ZIP file can be optionally retained for manual verification