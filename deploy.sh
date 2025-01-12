#!/bin/bash

# Default values
VERSION_TYPE="patch"  # Default version bump
BRANCH="main"         # Default branch

# Parse command-line arguments
while [[ "$1" =~ ^- ]]; do
  case "$1" in
    --minor)
      VERSION_TYPE="minor"
      shift
      ;;
    --patch)
      VERSION_TYPE="patch"
      shift
      ;;
    --major)
      VERSION_TYPE="major"
      shift
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Bump version
echo "Bumping version to $VERSION_TYPE"
poetry version "$VERSION_TYPE"

# Get the new version
VERSION=$(poetry version -s)

echo "Pushing tag to remote $BRANCH"
git push origin "dev:$BRANCH" --force

# Create a Git tag
echo "Creating Git tag v$VERSION"
git tag "v$VERSION"

# Push the tag to the remote
echo "Pushing tag to remote repository"
git push origin "v$VERSION"
