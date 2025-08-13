from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import cv2
import asyncio
import base64
from datetime import datetime, timedelta
from PIL import Image
import io
import numpy as np
from collections import defaultdict
import json
import traceback

# FIX: Point to parent directory where the modules are located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Captioning import predict_captions
from Yolo import detect_objects_yolo
from LLMs import useCohere
from main import caption_video, detect_scene_changes, caption_video_by_scenes

app = FastAPI(title="KAUST Vision Captioning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - static directory is in parent folder
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve frontend files directly from root
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend files not found. Please check static directory.</h1>", status_code=404)

@app.get("/style.css")
async def get_styles():
    css_path = os.path.join(STATIC_DIR, "css", "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path)
    return FileResponse(os.path.join(STATIC_DIR, "style.css"))  # fallback

@app.get("/app.js")
async def get_app_js():
    js_path = os.path.join(STATIC_DIR, "js", "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path)
    return FileResponse(os.path.join(STATIC_DIR, "app.js"))  # fallback

@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        print(f"Processing image upload: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # YOLO detection - Fixed: unpack all 3 values
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        raw_names, counts, annotated = detect_objects_yolo(cv_image)

        # Caption generation
        captions = predict_captions(
            [image],
            extra_info=[", ".join(raw_names)] if raw_names else None
        )

        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', annotated)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "caption": captions[0] if captions else "No caption generated",
            "detected_objects": raw_names,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
        }
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    temp_path = None
    try:
        print(f"Processing video upload: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Create temp file with proper extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.mp4'
        temp_path = os.path.join(
            BASE_DIR,
            f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"
        )
        
        # Read and save file
        contents = await file.read()
        print(f"Read {len(contents)} bytes from uploaded file")
        
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)
        
        print(f"Saved video to {temp_path}")
        
        # Verify file was written
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise Exception("Failed to save uploaded video file")

        # Process video
        print("Starting scene detection...")
        scenes = detect_scene_changes(temp_path, threshold=0.7)
        print(f"Detected {len(scenes)} scenes: {scenes[:5]}...")  # Show first 5
        
        # Limit scenes for processing speed
        limited_scenes = scenes[:10] if len(scenes) > 10 else scenes
        
        print("Starting caption generation...")
        captions = caption_video_by_scenes(temp_path, limited_scenes)
        print(f"Generated {len(captions)} captions")
        
        if not captions:
            # Fallback: try regular interval captioning
            print("No scene-based captions, trying interval-based...")
            captions = caption_video(temp_path, every_n_frames=60, batch_size=4)
        
        # Generate summary
        print("Generating summary...")
        summary = "No content to summarize"
        if captions:
            try:
                summary = useCohere(captions)
                print(f"Generated summary: {summary[:100]}...")
            except Exception as e:
                print(f"Summary generation failed: {e}")
                summary = f"Video processed successfully with {len(captions)} captions, but summary generation failed."

        result = {
            "success": True,
            "scenes": len(scenes),
            "captions": {str(k): v for k, v in captions.items()},
            "summary": summary,
            "frames_processed": len(captions),
            "video_duration": f"~{len(scenes) * 2}s estimated"  # rough estimate
        }
        
        print(f"Video processing complete. Result: {len(result['captions'])} captions, summary: {len(summary)} chars")
        return result
        
    except Exception as e:
        error_msg = f"Video processing error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"success": False, "error": error_msg}
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                print(f"Failed to cleanup {temp_path}: {e}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        # Store per-connection data for live summarization
        self.connection_data = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Initialize connection-specific data
        self.connection_data[id(websocket)] = {
            'captions_this_minute': {},
            'last_minute': datetime.now().replace(second=0, microsecond=0),
            'total_captions': 0
        }

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Cleanup connection data
        conn_id = id(websocket)
        if conn_id in self.connection_data:
            del self.connection_data[conn_id]

    async def send_summary(self, websocket: WebSocket, captions_dict):
        """Generate and send live summary"""
        try:
            if captions_dict:
                summary = useCohere(captions_dict)
                await websocket.send_json({
                    "type": "summary",
                    "summary": summary,
                    "caption_count": len(captions_dict)
                })
        except Exception as e:
            print(f"Failed to generate/send summary: {e}")

manager = ConnectionManager()

def find_working_camera():
    """Find first working camera index"""
    print("Searching for available cameras...")
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Found working camera at index {index}")
                return index
        cap.release()
    print("No working camera found")
    return None

@app.websocket("/ws/camera")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    cap = None
    frame_count = 0
    imgs_batch, meta_batch = [], []
    
    # Get connection-specific data
    conn_id = id(websocket)
    conn_data = manager.connection_data[conn_id]

    try:
        # Find working camera
        cam_index = find_working_camera()
        if cam_index is None:
            await websocket.send_json({"error": "No working camera found"})
            return

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            await websocket.send_json({"error": "Failed to open camera"})
            return

        print(f"WebSocket camera feed started for connection {conn_id}")

        while True:
            try:
                # Non-blocking receive with timeout
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                    if data.get("action") == "stop":
                        print("Received stop command")
                        break
                except asyncio.TimeoutError:
                    pass  # Continue with frame processing

                ret, frame = cap.read()
                if not ret:
                    await websocket.send_json({"error": "Failed to read camera frame"})
                    continue

                # Process every 15th frame for captioning, but always detect objects
                raw_names, counts, annotated = detect_objects_yolo(frame)
                
                caption = ""
                if frame_count % 15 == 0 and raw_names:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    imgs_batch.append(pil_img)
                    meta_batch.append(", ".join(raw_names))

                # Generate captions when batch is ready
                if len(imgs_batch) >= 3:
                    try:
                        captions = predict_captions(imgs_batch, extra_info=meta_batch)
                        if captions:
                            caption = captions[-1]
                            
                            # Store for live summarization
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            conn_data['captions_this_minute'].setdefault(timestamp, []).extend(captions)
                            conn_data['total_captions'] += len(captions)
                            
                    except Exception as e:
                        print(f"Caption error: {e}")
                    
                    imgs_batch.clear()
                    meta_batch.clear()

                # Check if a minute has passed for summarization
                current_minute = datetime.now().replace(second=0, microsecond=0)
                if (current_minute > conn_data['last_minute'] and 
                    conn_data['captions_this_minute'] and 
                    len(conn_data['captions_this_minute']) >= 3):  # Only summarize if we have enough data
                    
                    # Generate and send summary
                    await manager.send_summary(websocket, conn_data['captions_this_minute'])
                    
                    # Reset for next minute
                    conn_data['captions_this_minute'] = {}
                    conn_data['last_minute'] = current_minute

                # Encode and send frame
                _, buffer = cv2.imencode('.jpg', annotated)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                response_data = {
                    "frame": f"data:image/jpeg;base64,{frame_b64}",
                    "objects": raw_names,
                    "caption": caption,
                    "frame_count": frame_count,
                    "total_captions": conn_data['total_captions']
                }

                await websocket.send_json(response_data)
                frame_count += 1
                await asyncio.sleep(0.033)  # ~30 FPS

            except WebSocketDisconnect:
                print("WebSocket disconnected by client")
                break

    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"error": f"Camera error: {str(e)}"})
        except:
            pass

    finally:
        if cap:
            cap.release()
            print("Camera released")
        manager.disconnect(websocket)
        print(f"WebSocket connection {conn_id} closed")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "base_dir": BASE_DIR,
        "static_dir": STATIC_DIR,
        "static_exists": os.path.exists(STATIC_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting KAUST Vision Captioning System...")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATIC_DIR: {STATIC_DIR}")
    print(f"Static directory exists: {os.path.exists(STATIC_DIR)}")
    
    if os.path.exists(STATIC_DIR):
        print(f"Static files: {os.listdir(STATIC_DIR)}")
        print("Running on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    