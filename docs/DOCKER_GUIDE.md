# Docker Guide for Beginners 🐳

This guide will help you install and use Docker to run the LLM Cost Explorer, even if you've never used Docker before.

---

## What is Docker?

Think of Docker as a **shipping container for software**. Just like a shipping container can hold any cargo and be moved anywhere, a Docker container holds an application and everything it needs to run — and works the same on any computer.

**Why use Docker?**
- ✅ No need to install Python, libraries, or dependencies manually
- ✅ Works exactly the same on Windows, Mac, and Linux
- ✅ One command to run, one command to stop
- ✅ Doesn't interfere with other software on your computer

---

## Part 1: Installing Docker Desktop

### Windows

1. **Check System Requirements**
   - Windows 10 64-bit: Pro, Enterprise, or Education (Build 19041+)
   - Windows 11 64-bit: Any edition
   - WSL 2 enabled (Docker will help you set this up)

2. **Download Docker Desktop**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click **"Download for Windows"**

3. **Install**
   - Run the downloaded installer (`Docker Desktop Installer.exe`)
   - Follow the prompts (keep default options)
   - When prompted, ensure **"Use WSL 2"** is checked

4. **Restart Your Computer**
   - This is required for the installation to complete

5. **Start Docker Desktop**
   - Find "Docker Desktop" in your Start menu and open it
   - Wait for it to start (you'll see a whale icon in your system tray)
   - The first start may take a few minutes

6. **Verify Installation**
   - Open **Command Prompt** or **PowerShell**
   - Type: `docker --version`
   - You should see something like: `Docker version 24.0.6`

### Mac

1. **Check Your Mac's Chip**
   - Click the Apple menu → "About This Mac"
   - Note whether you have an **Intel** or **Apple Silicon (M1/M2/M3)** chip

2. **Download Docker Desktop**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click **"Download for Mac"**
   - Choose the version for your chip (Intel or Apple Silicon)

3. **Install**
   - Open the downloaded `.dmg` file
   - Drag Docker to your Applications folder
   - Open Docker from Applications

4. **Grant Permissions**
   - macOS will ask for permission to install networking components
   - Enter your password when prompted

5. **Wait for Docker to Start**
   - You'll see a whale icon in your menu bar
   - Wait until it stops animating (this means Docker is ready)

6. **Verify Installation**
   - Open **Terminal** (Applications → Utilities → Terminal)
   - Type: `docker --version`
   - You should see something like: `Docker version 24.0.6`

### Linux (Ubuntu/Debian)

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add your user to the docker group (so you don't need sudo)
sudo usermod -aG docker $USER

# Log out and back in for the group change to take effect
# Then verify:
docker --version
```

---

## Part 2: Docker Concepts (The Basics)

### Images vs Containers

| Concept | Analogy | Description |
|---------|---------|-------------|
| **Image** | Recipe | A blueprint with all instructions to build the app |
| **Container** | Cake | A running instance created from the image |

You can create many containers from one image, just like you can bake many cakes from one recipe.

### Key Commands

| Command | What it does |
|---------|--------------|
| `docker build` | Creates an image from a Dockerfile (like writing a recipe) |
| `docker run` | Starts a container from an image (like baking the cake) |
| `docker ps` | Shows running containers |
| `docker stop` | Stops a running container |
| `docker images` | Lists all images on your computer |

---

## Part 3: Running the LLM Cost Explorer

### Step 1: Get the Code

**Option A: Download ZIP**
1. Go to https://github.com/dlwhyte/AgenticAI_foundry
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file to a folder you'll remember (e.g., `Documents/AgenticAI_foundry`)

**Option B: Clone with Git** (if you have Git installed)
```bash
git clone https://github.com/dlwhyte/AgenticAI_foundry.git
```

### Step 2: Open a Terminal

**Windows:**
- Press `Win + R`, type `cmd`, press Enter
- Or search for "Command Prompt" in the Start menu
- Or search for "PowerShell" (recommended)

**Mac:**
- Press `Cmd + Space`, type "Terminal", press Enter
- Or go to Applications → Utilities → Terminal

**Linux:**
- Press `Ctrl + Alt + T`
- Or search for "Terminal" in your applications

### Step 3: Navigate to the Project Folder

```bash
# Windows (Command Prompt)
cd C:\Users\YourName\Documents\AgenticAI_foundry

# Windows (PowerShell) - same as above
cd C:\Users\YourName\Documents\AgenticAI_foundry

# Mac/Linux
cd ~/Documents/AgenticAI_foundry
```

**Tip:** You can drag the folder onto the terminal window to paste the path!

### Step 4: Build the Docker Image

This creates the image (downloads dependencies, sets everything up). Only needed once.

```bash
docker build -t agenticai-foundry .
```

- `-t agenticai-foundry` gives the image a name
- `.` tells Docker to look in the current folder for the Dockerfile

**This will take 2-3 minutes** the first time (it's downloading Python and libraries).

You'll see lots of output. Wait until you see:
```
Successfully built xxxxxxxxxx
Successfully tagged agenticai-foundry:latest
```

### Step 5: Run the Container

```bash
docker run -p 8501:8501 agenticai-foundry
```

- `-p 8501:8501` connects port 8501 on your computer to port 8501 in the container

You'll see output like:
```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

### Step 6: Open the App

Open your web browser and go to:
```
http://localhost:8501
```

🎉 **You should see the LLM Cost Explorer!**

### Step 7: Stop the App

When you're done, go back to the terminal and press:
```
Ctrl + C
```

This stops the container.

---

## Part 4: Common Issues & Solutions

### "Docker command not found"
- **Cause:** Docker isn't installed or terminal was opened before installation
- **Solution:** 
  1. Make sure Docker Desktop is installed
  2. Close and reopen your terminal
  3. On Windows, try restarting your computer

### "Cannot connect to Docker daemon"
- **Cause:** Docker Desktop isn't running
- **Solution:** 
  1. Open Docker Desktop application
  2. Wait for the whale icon to stop animating
  3. Try the command again

### "Port 8501 is already in use"
- **Cause:** Another app (or another container) is using that port
- **Solution:** Use a different port:
  ```bash
  docker run -p 8502:8501 agenticai-foundry
  ```
  Then open `http://localhost:8502` instead

### "Error response from daemon: pull access denied"
- **Cause:** Trying to pull an image that doesn't exist
- **Solution:** Make sure you're building, not pulling:
  ```bash
  docker build -t agenticai-foundry .
  ```

### Build is very slow
- **Cause:** First-time download of Python and libraries (~500MB)
- **Solution:** This is normal for the first build. Subsequent builds are much faster due to caching.

### "COPY failed: file not found"
- **Cause:** You're not in the right directory
- **Solution:** Make sure you `cd` into the folder containing the `Dockerfile`:
  ```bash
  cd path/to/AgenticAI_foundry
  ls  # Should show Dockerfile, Home.py, etc.
  ```

### App works but looks different than expected
- **Cause:** Browser cache
- **Solution:** Hard refresh with `Ctrl + Shift + R` (Windows/Linux) or `Cmd + Shift + R` (Mac)

---

## Part 5: Useful Docker Commands Reference

```bash
# See running containers
docker ps

# See all containers (including stopped)
docker ps -a

# Stop a container
docker stop <container_id>

# Remove a container
docker rm <container_id>

# See all images
docker images

# Remove an image
docker rmi agenticai-foundry

# Rebuild without cache (if you need a fresh build)
docker build --no-cache -t agenticai-foundry .

# Run in background (detached mode)
docker run -d -p 8501:8501 agenticai-foundry

# View logs of a background container
docker logs <container_id>

# Stop all running containers
docker stop $(docker ps -q)
```

---

## Part 6: Cleanup

If you want to free up disk space later:

```bash
# Remove the container (after stopping it)
docker rm $(docker ps -a -q --filter ancestor=agenticai-foundry)

# Remove the image
docker rmi agenticai-foundry

# Remove all unused Docker data (careful!)
docker system prune
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│                    DOCKER QUICK START                    │
├─────────────────────────────────────────────────────────┤
│  1. Open terminal                                        │
│  2. cd AgenticAI_foundry                                │
│  3. docker build -t agenticai-foundry .                 │
│  4. docker run -p 8501:8501 agenticai-foundry           │
│  5. Open http://localhost:8501                          │
│  6. Ctrl+C to stop                                       │
└─────────────────────────────────────────────────────────┘
```

---

## Need More Help?

- **Docker Documentation:** https://docs.docker.com/get-started/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **Course Discussion Forum:** Post your question with any error messages

---

*MIT Professional Education: Agentic AI*
