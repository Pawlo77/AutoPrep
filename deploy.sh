#!/bin/bash

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
