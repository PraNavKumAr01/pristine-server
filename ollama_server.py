from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import uvicorn
from typing import Optional
import os
import platform
import subprocess
import requests
import time
import psutil
from pathlib import Path
import atexit

windows_url = "https://ollama.com/download/OllamaSetup.exe"
mac_url = "https://ollama.com/download/Ollama-darwin.zip"
linx_url = "curl -fsSL https://ollama.com/install.sh | sh"

app = FastAPI()

class OllamaManager:
    def __init__(self, llm_name, install_path=None):
        self.model_name = llm_name
        self.system = platform.system().lower()
        self.install_path = install_path or self._get_default_install_path()
        self.server_process = None
        atexit.register(self.stop_server)

    def _get_default_install_path(self):
        if self.system == "windows":
            return Path(os.environ.get("LOCALAPPDATA")) / "Ollama"
        elif self.system == "darwin":
            return Path.home() / "Library" / "Application Support" / "Ollama"
        else: 
            return Path.home() / ".ollama"

    def _get_ollama_executable(self):
        """Get the path to the Ollama executable based on OS"""
        if self.system == "windows":
            return str(self.install_path / "ollama.exe")
        elif self.system == "darwin":
            return "/usr/local/bin/ollama"
        else:
            return "/usr/local/bin/ollama"

    def install_ollama(self):
        """Install Ollama based on the operating system"""
        if self.system == "windows":
            return self._install_windows()
        elif self.system == "darwin":
            return self._install_macos()
        else:
            return self._install_linux()

    def _install_windows(self):
        ollama_url = windows_url
        installer_path = self.install_path / "ollama-installer.exe"
        
        self.install_path.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(ollama_url)
        with open(installer_path, 'wb') as f:
            f.write(response.content)
        
        subprocess.run([str(installer_path), '/VERYSILENT'], check=True)
        return True

    def _install_macos(self):
        try:
            subprocess.run(['brew', 'install', 'ollama'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Error: Failed to install Ollama via Homebrew")
            return False

    def _install_linux(self):
        try:
            subprocess.run('curl https://ollama.ai/install.sh | sh', shell=True, check=True)
            return True
        except subprocess.CalledProcessError:
            print("Error: Failed to install Ollama")
            return False

    def is_server_running(self):
        """Check if the Ollama server is running"""
        try:
            import socket
            with socket.create_connection(("localhost", 11434), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout):
            return False

    def start_server(self):
        """Start the Ollama server if it's not already running"""
        if self.is_server_running():
            print("Ollama server is already running")
            return True

        try:
            if self.system == "windows":
                self.server_process = subprocess.Popen(
                    [self._get_ollama_executable(), "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                self.server_process = subprocess.Popen(
                    [self._get_ollama_executable(), "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True
                )

            max_attempts = 10
            attempts = 0
            while attempts < max_attempts:
                if self.is_server_running():
                    print("Ollama server started successfully")
                    return True
                time.sleep(1)
                attempts += 1

            if not self.is_server_running():
                print("Failed to start Ollama server")
                return False

            return True

        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            return False

    def stop_server(self):
        """Stop the Ollama server"""
        if self.server_process:
            try:
                if self.system == "windows":
                    parent = psutil.Process(self.server_process.pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                else:
                    self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("Ollama server stopped")
            except Exception as e:
                print(f"Error stopping Ollama server: {e}")
        
        self._cleanup_remaining_processes()

    def _cleanup_remaining_processes(self):
        """Clean up any remaining Ollama processes"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def pull_model(self):
        """Pull the specified model using Ollama CLI"""
        try:
            subprocess.run([self._get_ollama_executable(), 'pull', self.model_name], check=True)
            return True
        except subprocess.CalledProcessError:
            print(f"Error: Failed to pull model {self.model_name}")
            return False

    def verify_installation(self):
        """Verify Ollama installation and model availability"""
        try:
            subprocess.run([self._get_ollama_executable(), 'list'], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

class ModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    llm_name: str
    install_path: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    message: str

# # Add CORS middleware to allow requests from your Electron app
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, replace with your Electron app's origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

manager = None

@app.post("/init", response_model=StatusResponse)
async def initialize_ollama(request: ModelRequest):
    global manager
    try:
        manager = OllamaManager(llm_name=request.llm_name, install_path=request.install_path)
        return StatusResponse(status="success", message="Ollama manager initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/install", response_model=StatusResponse)
async def install_ollama():
    if not manager:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /init first")
    
    try:
        if manager.verify_installation():
            return StatusResponse(status="success", message="Ollama is already installed")
        
        if manager.install_ollama():
            return StatusResponse(status="success", message="Ollama installed successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to install Ollama")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start", response_model=StatusResponse)
async def start_server():
    if not manager:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /init first")
    
    try:
        if manager.start_server():
            return StatusResponse(status="success", message="Server started successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to start server")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop", response_model=StatusResponse)
async def stop_server():
    if not manager:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /init first")
    
    try:
        manager.stop_server()
        return StatusResponse(status="success", message="Server stopped successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pull-model", response_model=StatusResponse)
async def pull_model():
    if not manager:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /init first")
    
    try:
        if manager.pull_model():
            return StatusResponse(status="success", message=f"Model {manager.model_name} pulled successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to pull model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_status():
    if not manager:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /init first")
    
    try:
        is_running = manager.is_server_running()
        return StatusResponse(
            status="running" if is_running else "stopped",
            message="Server is running" if is_running else "Server is not running"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)