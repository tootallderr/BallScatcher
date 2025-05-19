#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
from datetime import datetime
from time import sleep
import platform
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set up logging
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("sync_log.txt", mode='a', encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger("github_sync")

# CONFIG
REPO_PATH = os.path.abspath(os.path.dirname(__file__))
REMOTE_URL = "https://github.com/tootallderr/BallScatcher.git"  # Replace with your actual remote URL
# Set the commit message to include a timestamp
COMMIT_MESSAGE = f"📝 Auto-sync on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
# Don't hardcode the branch name, let's detect it dynamically
DEFAULT_BRANCH = "master"  # Changed from "main" to match your actual branch
# Path to log file
LOG_FILE_PATH = os.path.join(REPO_PATH, "sync_log.txt")

# Add configuration for Git settings
GIT_CONFIG = {
    "core.autocrlf": "true",       # Handle line endings automatically
    "diff.renameLimit": "10000",   # Increase rename detection limit
    "diff.renames": "true"         # Ensure rename detection is enabled
}

class GitSyncException(Exception):
    """Custom exception for Git sync errors"""
    pass

def run_command(command, cwd, check_error=True, silent=False):
    """Run a shell command and return result with improved error handling"""
    if not silent:
        logger.info(f"Running: {command}")
        
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if not silent and result.stdout:
            logger.info(result.stdout.strip())
            
        if result.stderr:
            # Don't treat warnings as errors
            if ("error:" in result.stderr.lower() and "warning:" not in result.stderr.lower()) or result.returncode != 0:
                logger.error(f"❌ {result.stderr.strip()}")
                if check_error:
                    raise GitSyncException(result.stderr.strip())
            elif "warning:" in result.stderr.lower():
                logger.warning(f"⚠️ {result.stderr.strip()}")
            elif not silent:
                logger.info(result.stderr.strip())
                
        return result
    except Exception as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {str(e)}")
        if check_error:
            raise GitSyncException(f"Command failed: {str(e)}")
        return None

def get_current_branch(cwd):
    """Get the current Git branch."""
    result = run_command("git branch --show-current", cwd, silent=True)
    if result and result.returncode == 0:
        branch = result.stdout.strip()
        if branch:
            return branch
            
    # Fallback to more robust method if the above doesn't work
    result = run_command("git rev-parse --abbrev-ref HEAD", cwd, silent=True)
    if result and result.returncode == 0:
        branch = result.stdout.strip()
        if branch and branch != "HEAD":
            return branch
            
    return DEFAULT_BRANCH

def apply_git_configs():
    """Apply Git configurations to avoid warnings and improve performance."""
    logger.info("🔧 Applying Git configurations...")
    for key, value in GIT_CONFIG.items():
        run_command(f'git config {key} {value}', REPO_PATH, silent=True)

def has_changes(cwd):
    """Check if there are changes to commit."""
    # Check for staged changes
    staged = run_command("git diff --cached --exit-code", cwd, check_error=False, silent=True)
    # Check for unstaged changes
    unstaged = run_command("git diff --exit-code", cwd, check_error=False, silent=True)
    # Check for untracked files
    untracked = run_command("git ls-files --others --exclude-standard", cwd, check_error=False, silent=True)
    
    return (staged and staged.returncode != 0) or \
           (unstaged and unstaged.returncode != 0) or \
           (untracked and untracked.stdout.strip())

def check_git_installed():
    """Check if Git is installed on the system."""
    try:
        run_command("git --version", REPO_PATH, silent=True)
        return True
    except GitSyncException:
        logger.error("❌ Git is not installed or not in PATH. Please install Git.")
        return False

def check_network_connectivity():
    """Check if we can reach GitHub."""
    cmd = "ping -c 1 github.com" if platform.system() != "Windows" else "ping -n 1 github.com"
    try:
        result = run_command(cmd, REPO_PATH, check_error=False, silent=True)
        return result and result.returncode == 0
    except:
        return False

def check_remote_exists():
    """Check if the remote repository exists and is accessible."""
    try:
        result = run_command(f"git ls-remote {REMOTE_URL}", REPO_PATH, check_error=False, silent=True)
        return result and result.returncode == 0
    except:
        return False

def check_auth_setup():
    """Check if GitHub authentication is set up."""
    try:
        result = run_command("git config --get credential.helper", REPO_PATH, check_error=False, silent=True)
        return result and result.stdout.strip()
    except:
        return False

def check_and_create_gitignore():
    """Check for .gitignore file and create a default one if it doesn't exist."""
    gitignore_path = os.path.join(REPO_PATH, ".gitignore")
    
    if not os.path.exists(gitignore_path):
        logger.info("🔧 Creating default .gitignore file...")
        
        # Common Python project ignores
        default_ignores = [
            "# Python",
            "__pycache__/",
            "*.py[cod]",
            "*$py.class",
            "*.so",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "MANIFEST",
            "",
            "# Environments",
            ".env",
            ".venv",
            "env/",
            "venv/",
            "ENV/",
            "env.bak/",
            "venv.bak/",
            "",
            "# Logs",
            "*.log",
            "sync_log.txt",
            "",
            "# IDE files",
            ".idea/",
            ".vscode/",
            "*.swp",
            "*.swo",
            "",
            "# OS specific",
            ".DS_Store",
            "Thumbs.db",
            "",
            "# Project specific",
            "backend/data/temp/",
            "backend/data/cache/",
            "node_modules/",
            "package-lock.json",
        ]
        
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_ignores))
        
        logger.info("✅ Created default .gitignore file")
        return True
    
    return False

def delete_log_file():
    """Delete the sync log file if it exists."""
    try:
        if os.path.exists(LOG_FILE_PATH):
            os.remove(LOG_FILE_PATH)
            print(f"✅ Deleted sync log file: {LOG_FILE_PATH}")
            return True
    except Exception as e:
        print(f"⚠️ Could not delete log file: {str(e)}")
        return False
    return False

def sync_to_github():
    """Sync local repository to GitHub with improved error handling."""
    logger.info("🚀 Starting sync process...\n")

    try:
        # Check prerequisites
        if not check_git_installed():
            return False

        if not check_network_connectivity():
            logger.error("❌ Cannot connect to GitHub. Check your internet connection.")
            return False

        # Initialize repository if needed
        if not os.path.exists(os.path.join(REPO_PATH, ".git")):
            logger.info("🔧 Initializing Git repository...")
            run_command("git init", REPO_PATH)

            if not check_remote_exists():
                logger.error(f"❌ Remote repository {REMOTE_URL} doesn't exist or is not accessible.")
                return False

            run_command(f"git remote add origin {REMOTE_URL}", REPO_PATH)

            # Create .gitignore on first init if it doesn't exist
            check_and_create_gitignore()
            
            # Apply git configs after repo is initialized
            apply_git_configs()
        else:
            # Check if remote is correctly set
            remote_check = run_command("git remote -v", REPO_PATH, silent=True)
            if REMOTE_URL not in remote_check.stdout:
                logger.info("🔄 Updating remote URL...")
                run_command("git remote remove origin", REPO_PATH, check_error=False)
                run_command(f"git remote add origin {REMOTE_URL}", REPO_PATH)
            
            # Apply git configs to existing repo
            apply_git_configs()
            check_repo_size()

        # Check for .gitignore file and create if needed
        gitignore_created = check_and_create_gitignore()
        if gitignore_created:
            logger.info("📝 Adding .gitignore to repository...")
            run_command("git add .gitignore", REPO_PATH)
        
        # Get current branch or use default
        current_branch = get_current_branch(REPO_PATH)
        logger.info(f"🔍 Using branch: {current_branch}")
        
        # Check if .gitignore is being respected
        run_command("git check-ignore --no-index --verbose .", REPO_PATH, check_error=False, silent=True)
        
        # Stage all files - respecting .gitignore
        logger.info("📦 Staging files (respecting .gitignore)...")
        # Stage in smaller batches if needed for large repos
        try:
            run_command("git add .", REPO_PATH)  # Changed from -A to . to respect .gitignore
        except GitSyncException:
            logger.warning("⚠️ Failed to add all files at once, trying batch mode...")
            # Alternative approach for large repos - add by directory
            for root, dirs, _ in os.walk(REPO_PATH):
                if ".git" in root:
                    continue
                # Skip directories that might be in .gitignore
                if any(d for d in [".env", "venv", "node_modules", "__pycache__"] if d in dirs):
                    dirs[:] = [d for d in dirs if d not in [".env", "venv", "node_modules", "__pycache__"]]
                rel_path = os.path.relpath(root, REPO_PATH)
                if rel_path != ".":
                    run_command(f'git add "{rel_path}"', REPO_PATH, check_error=False)
        
        # Check if there are changes to commit
        if not has_changes(REPO_PATH):
            logger.info("✅ No changes to commit")
            return True
        
        # Create initial commit if needed
        has_any_commits = run_command("git rev-parse --verify HEAD", REPO_PATH, check_error=False, silent=True)
        if not has_any_commits or has_any_commits.returncode != 0:
            logger.info("📝 Creating initial commit...")
            run_command('git commit -m "📦 Initial commit"', REPO_PATH)
            
            # For initial commit, don't try to pull since the branch doesn't exist remotely yet
            logger.info("⬆️ Pushing initial commit to GitHub...")
            run_command(f"git push --set-upstream origin {current_branch}", REPO_PATH, check_error=False)
        else:
            # Commit changes
            logger.info("📝 Committing changes...")
            run_command(f'git commit -m "{COMMIT_MESSAGE}"', REPO_PATH)
            
            # Check if branch exists on remote before pulling
            if check_remote_branch(current_branch):
                # Pull before pushing to avoid conflicts
                logger.info("⬇️ Pulling latest changes from remote...")
                pull_result = run_command(f"git pull --no-edit origin {current_branch}", REPO_PATH, check_error=False)
                
                # Check for merge conflicts
                if pull_result and "CONFLICT" in pull_result.stdout + pull_result.stderr:
                    logger.error("❌ Merge conflicts detected! Resolve conflicts manually before syncing.")
                    run_command("git merge --abort", REPO_PATH, check_error=False)
                    return False
            else:
                # Branch doesn't exist remotely yet, don't try to pull
                # This message is already logged in check_remote_branch()
                pass
            
            # Push changes
            logger.info("⬆️ Pushing to GitHub...")
            push_result = run_command(f"git push origin {current_branch}", REPO_PATH, check_error=False)
            
            # Handle authentication issues
            if push_result and push_result.returncode != 0:
                if "Authentication failed" in push_result.stderr:
                    logger.error("❌ GitHub authentication failed!")
                    if not check_auth_setup():
                        logger.info("💡 Tip: Set up credential helper with: git config --global credential.helper cache")
                    return False
                    
                # Try pushing with --set-upstream for new branches
                if "has no upstream branch" in push_result.stderr:
                    logger.info(f"🔄 Setting upstream for branch {current_branch}...")
                    run_command(f"git push --set-upstream origin {current_branch}", REPO_PATH)
                else:
                    logger.error(f"❌ Push failed: {push_result.stderr.strip()}")
                    return False
                
        logger.info("\n✅ Sync to GitHub complete!")
        return True
        
    except GitSyncException as e:
        logger.error(f"❌ Sync failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return False

def cleanup_git_repo():
    """Clean up Git repository if it's in a bad state."""
    logger.info("🧹 Cleaning up Git repository...")
    
    # Clean up garbage objects
    run_command("git gc --prune=now", REPO_PATH, check_error=False, silent=True)
    
    # Remove any index.lock file if it exists (from interrupted git operations)
    git_lock_file = os.path.join(REPO_PATH, ".git", "index.lock")
    if os.path.exists(git_lock_file):
        try:
            os.remove(git_lock_file)
            logger.info("✅ Removed stale .git/index.lock file")
        except Exception as e:
            logger.warning(f"⚠️ Could not remove lock file: {str(e)}")
    
    return True

def check_repo_size():
    """Check if repository is getting too large."""
    try:
        result = run_command("git count-objects -v", REPO_PATH, silent=True)
        if result and result.returncode == 0:
            # Extract size-pack value (in KB)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('size-pack:'):
                    size_kb = int(line.split(':')[1].strip())
                    size_mb = size_kb / 1024
                    if size_mb > 500:  # Warning if over 500MB
                        logger.warning(f"⚠️ Repository size is large: {size_mb:.2f} MB")
                        logger.warning("Consider using Git LFS for large binary files: https://git-lfs.github.com")
    except:
        pass  # Silently fail if we can't check size

def get_gitignore_status():
    """Get a report on what files are being ignored by .gitignore."""
    try:
        # Get list of ignored files
        ignored_files = run_command("git status --ignored --porcelain", REPO_PATH, check_error=False, silent=True)
        if ignored_files and ignored_files.stdout:
            ignored_list = []
            for line in ignored_files.stdout.splitlines():
                if line.startswith("!!"):
                    ignored_list.append(line[3:])
            
            if ignored_list:
                logger.info(f"ℹ️ {len(ignored_list)} files/directories are being ignored by .gitignore")
                if len(ignored_list) <= 5:
                    for item in ignored_list:
                        logger.info(f"   - {item}")
                else:
                    for item in ignored_list[:3]:
                        logger.info(f"   - {item}")
                    logger.info(f"   - ... and {len(ignored_list) - 3} more")
    except Exception as e:
        logger.error(f"❌ Error checking .gitignore: {str(e)}")

def check_remote_branch(branch_name):
    """Check if branch exists on remote and handle correctly."""
    # Try direct check first (fast)
    result = run_command(f"git ls-remote --heads origin {branch_name}", REPO_PATH, check_error=False, silent=True)
    branch_exists = result and result.returncode == 0 and result.stdout.strip() != ""
    
    if branch_exists:
        logger.info(f"✅ Remote branch '{branch_name}' exists")
        return True
    else:
        logger.info(f"ℹ️ Remote branch '{branch_name}' doesn't exist yet - will create on push")
        return False

def list_remote_branches():
    """Get a list of all known remote branches."""
    # First, make sure we have latest remote info
    run_command("git fetch --prune", REPO_PATH, check_error=False, silent=True)
    
    result = run_command("git branch -r", REPO_PATH, check_error=False, silent=True)
    if result and result.returncode == 0:
        branches = []
        for line in result.stdout.splitlines():
            branch = line.strip()
            if branch.startswith("origin/"):
                branches.append(branch.replace("origin/", ""))
        return branches
    return []

if __name__ == "__main__":
    try:
        # Install tqdm if needed
        try:
            from tqdm import tqdm
            
            # Run the sync process (moved before progress display)
            sync_result = sync_to_github()
            
            # Use tqdm for visual feedback after the fact
            steps = [
                "Prerequisites checked",
                "Repository prepared",
                "Files staged",
                "Changes committed",
                "Remote synced",
                "Push completed"
            ]
            
            with tqdm(steps, desc="🔄 Completed", ncols=80) as progress:
                for _ in progress:
                    sleep(0.1)
            
            # If sync was successful, show gitignore status
            if sync_result:
                get_gitignore_status()
                # Delete log file after successful sync
                delete_log_file()
            
        except ImportError:
            logger.warning("⚠️ tqdm not installed, running without progress bar")
            sync_result = sync_to_github()
            if sync_result:
                get_gitignore_status()
                # Delete log file after successful sync
                delete_log_file()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Sync interrupted by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {str(e)}")