// KAUST Vision Captioning System - Enhanced Frontend JavaScript

class VisionCaptioningApp {
    constructor() {
        this.currentMode = 'upload';
        this.websocket = null;
        this.isProcessing = false;
        this.liveSummaries = [];

        this.initializeEventListeners();
        this.initializeDragAndDrop();
    }

    initializeEventListeners() {
        // Mode switching
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.closest('.mode-btn').dataset.mode;
                this.switchMode(mode);
            });
        });

        // File inputs
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleImageUpload(e));
        }

        const videoInput = document.getElementById('video-input');
        if (videoInput) {
            videoInput.addEventListener('change', (e) => this.handleVideoUpload(e));
        }

        // Camera controls
        const startBtn = document.getElementById('start-camera');
        if (startBtn) startBtn.addEventListener('click', () => this.startCamera());
        const stopBtn = document.getElementById('stop-camera');
        if (stopBtn) stopBtn.addEventListener('click', () => this.stopCamera());

        // Upload zones
        const uploadZone = document.getElementById('upload-zone');
        if (uploadZone) {
            uploadZone.addEventListener('click', () => {
                const el = document.getElementById('file-input');
                if (el) el.click();
            });
        }

        const videoUploadZone = document.getElementById('video-upload-zone');
        if (videoUploadZone) {
            videoUploadZone.addEventListener('click', () => {
                const el = document.getElementById('video-input');
                if (el) el.click();
            });
        }

        // Reset upload
        const resetBtn = document.getElementById('reset-upload');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetUpload());
        }
    }

    initializeDragAndDrop() {
        // Image upload drag and drop
        const uploadZone = document.getElementById('upload-zone');
        this.setupDragAndDrop(uploadZone, (files) => {
            if (files[0] && files[0].type.startsWith('image/')) {
                this.processImageFile(files[0]);
            }
        });

        // Video upload drag and drop
        const videoUploadZone = document.getElementById('video-upload-zone');
        this.setupDragAndDrop(videoUploadZone, (files) => {
            if (files[0] && files[0].type.startsWith('video/')) {
                this.processVideoFile(files[0]);
            }
        });
    }

    setupDragAndDrop(element, callback) {
        if (!element) return;
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            element.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            element.addEventListener(eventName, () => {
                element.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            element.addEventListener(eventName, () => {
                element.classList.remove('dragover');
            });
        });

        element.addEventListener('drop', (e) => {
            const files = Array.from(e.dataTransfer.files);
            callback(files);
        });
    }

    switchMode(mode) {
        if (this.isProcessing) {
            this.showToast('Please wait for the current process to complete', 'warning');
            return;
        }
        if (!mode) return;

        // Update active button
        document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.querySelector(`[data-mode="${mode}"]`);
        if (activeBtn) activeBtn.classList.add('active');

        // Update content visibility
        document.querySelectorAll('.mode-content').forEach(content => content.classList.remove('active'));
        const activeContent = document.getElementById(`${mode}-mode`);
        if (activeContent) activeContent.classList.add('active');

        // Stop camera if switching away from camera mode
        if (this.currentMode === 'camera' && mode !== 'camera') {
            this.stopCamera();
        }

        this.currentMode = mode;
        this.clearResults();
    }

    // Image Upload Handling
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) this.processImageFile(file);
    }

    async processImageFile(file) {
        if (this.isProcessing) return;

        // Show preview
        this.showImagePreview(file);

        // Upload and process
        this.isProcessing = true;
        this.showLoading('Processing image...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayImageResults(result);
                this.showToast('Image processed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Unknown server error');
            }
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
            console.error('Image processing error:', error);
        } finally {
            this.hideLoading();
            this.isProcessing = false;
        }
    }

    showImagePreview(file) {
        const preview = document.getElementById('image-preview');
        const img = document.getElementById('preview-img');
        const uploadZone = document.getElementById('upload-zone');

        if (!preview || !img || !uploadZone) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target.result;
            uploadZone.style.display = 'none';
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    displayImageResults(result) {
        const resultsContent = document.getElementById('results-content');
        if (!resultsContent) return;

        let objectTags = '';
        if (result.detected_objects && result.detected_objects.length > 0) {
            objectTags = result.detected_objects.map(obj =>
                `<span class="object-tag">${obj}</span>`
            ).join('');
        }

        resultsContent.innerHTML = `
            <div class="caption-result">
                <h6><i class="fas fa-comment-alt"></i> Generated Caption</h6>
                <p>${result.caption}</p>
                ${objectTags ? `
                    <h6 class="mt-3"><i class="fas fa-tags"></i> Detected Objects</h6>
                    <div class="detected-objects">${objectTags}</div>
                ` : ''}
            </div>
        `;

        // Update preview with annotated image if available
        if (result.annotated_image) {
            const img = document.getElementById('preview-img');
            if (img) img.src = result.annotated_image;
        }
    }

    // Camera Handling
    async startCamera() {
        if (this.isProcessing) return;

        try {
            this.isProcessing = true;
            this.liveSummaries = []; // Reset summaries
            this.showLoading('Starting camera...');

            // Connect WebSocket
            const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.websocket = new WebSocket(`${wsProtocol}//${location.host}/ws/camera`);

            this.websocket.onopen = () => {
                this.hideLoading();
                this.isProcessing = false;
                const startBtn = document.getElementById('start-camera');
                const stopBtn = document.getElementById('stop-camera');
                const placeholder = document.querySelector('.camera-placeholder');
                const feed = document.getElementById('camera-feed');

                if (startBtn) startBtn.style.display = 'none';
                if (stopBtn) stopBtn.style.display = 'inline-block';
                if (placeholder) placeholder.style.display = 'none';
                if (feed) feed.style.display = 'block';

                this.showToast('Camera started successfully!', 'success');

                // Start the camera feed
                this.websocket.send(JSON.stringify({ action: 'start' }));
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.error) {
                    this.showToast(`Camera error: ${data.error}`, 'error');
                    this.stopCamera();
                    return;
                }

                // Handle different message types
                if (data.type === 'summary') {
                    // Live summary received
                    this.liveSummaries.push({
                        time: new Date().toLocaleTimeString(),
                        summary: data.summary,
                        caption_count: data.caption_count
                    });
                    this.showToast(`Live summary generated from ${data.caption_count} captions!`, 'info');
                }

                // Update camera feed
                if (data.frame) {
                    const feed = document.getElementById('camera-feed');
                    if (feed) feed.src = data.frame;
                }

                // Update results with both current data and summaries
                this.displayCameraResults(data);
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showToast('Camera connection error', 'error');
                this.stopCamera();
            };

            this.websocket.onclose = () => {
                this.stopCamera();
            };
        } catch (error) {
            this.hideLoading();
            this.isProcessing = false;
            this.showToast(`Camera error: ${error.message}`, 'error');
            console.error('Camera start error:', error);
        }
    }

    stopCamera() {
        if (this.websocket) {
            try {
                if (this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(JSON.stringify({ action: 'stop' }));
                }
            } catch {}
            this.websocket.close();
            this.websocket = null;
        }

        const startBtn = document.getElementById('start-camera');
        const stopBtn = document.getElementById('stop-camera');
        const placeholder = document.querySelector('.camera-placeholder');
        const feed = document.getElementById('camera-feed');

        if (startBtn) startBtn.style.display = 'inline-block';
        if (stopBtn) stopBtn.style.display = 'none';
        if (placeholder) placeholder.style.display = 'block';
        if (feed) feed.style.display = 'none';

        this.clearResults();
        this.isProcessing = false;
    }

    displayCameraResults(data) {
        const resultsContent = document.getElementById('results-content');
        if (!resultsContent) return;

        let content = '';

        // Show live summaries first (most important)
        if (this.liveSummaries.length > 0) {
            const latestSummaries = this.liveSummaries.slice(-3); // Show last 3 summaries
            content += `
                <div class="summary-section mb-3">
                    <h6><i class="fas fa-brain"></i> Live Summaries</h6>
                    ${latestSummaries.map(summary => `
                        <div class="caption-item mb-2 p-2" style="background: rgba(0, 167, 157, 0.1); border-radius: 8px;">
                            <small class="text-muted">${summary.time} (${summary.caption_count} captions):</small><br>
                            <strong>${summary.summary}</strong>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Show current caption
        if (data.caption) {
            content += `
                <div class="caption-result">
                    <h6><i class="fas fa-video"></i> Latest Caption</h6>
                    <p>${data.caption}</p>
                </div>
            `;
        }

        // Show currently detected objects
        if (data.objects && data.objects.length > 0) {
            const objectTags = data.objects.map(obj =>
                `<span class="object-tag">${obj}</span>`
            ).join('');

            content += `
                <div class="mt-3">
                    <h6><i class="fas fa-search"></i> Currently Detecting</h6>
                    <div class="detected-objects">${objectTags}</div>
                </div>
            `;
        }

        // Show stats
        if (data.total_captions !== undefined) {
            content += `
                <div class="mt-3">
                    <small class="text-muted">
                        <i class="fas fa-chart-line"></i> 
                        Total captions: ${data.total_captions} | 
                        Summaries: ${this.liveSummaries.length}
                    </small>
                </div>
            `;
        }

        if (content) {
            resultsContent.innerHTML = content;
        }
    }

    // Video Upload Handling
    handleVideoUpload(event) {
        const file = event.target.files[0];
        if (file) this.processVideoFile(file);
    }

    async processVideoFile(file) {
        if (this.isProcessing) return;

        this.isProcessing = true;
        this.showVideoProcessing();
        this.showLoading('Processing video... This may take a while.');

        try {
            console.log('Starting video upload:', file.name, 'Size:', file.size);
            
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload-video', {
                method: 'POST',
                body: formData
            });

            console.log('Upload response status:', response.status);
            const result = await response.json();
            console.log('Upload result:', result);

            if (result.success) {
                this.displayVideoResults(result);
                this.showToast('Video processed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Unknown server error');
            }
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
            console.error('Video processing error:', error);
        } finally {
            this.hideVideoProcessing();
            this.hideLoading();
            this.isProcessing = false;
        }
    }

    showVideoProcessing() {
        const zone = document.getElementById('video-upload-zone');
        const processing = document.getElementById('video-processing');
        if (zone) zone.style.display = 'none';
        if (processing) processing.style.display = 'block';

        // Simulate progress
        const progressBar = document.querySelector('.glass-progress');
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 85) progress = 85; // Don't go to 100% until done
            if (progressBar) progressBar.style.width = `${progress}%`;
        }, 2000);

        this.progressInterval = interval;
    }

    hideVideoProcessing() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        const progressBar = document.querySelector('.glass-progress');
        if (progressBar) progressBar.style.width = '100%';
        
        setTimeout(() => {
            const processing = document.getElementById('video-processing');
            const zone = document.getElementById('video-upload-zone');
            if (processing) processing.style.display = 'none';
            if (zone) zone.style.display = 'block';
            if (progressBar) progressBar.style.width = '0%';
        }, 1000);
    }

    displayVideoResults(result) {
        const resultsContent = document.getElementById('results-content');
        if (!resultsContent) return;

        console.log('Displaying video results:', result);

        // Show summary prominently
        let content = '';
        if (result.summary) {
            content += `
                <div class="summary-section mb-3">
                    <h6><i class="fas fa-film"></i> Video Summary</h6>
                    <div class="summary-box p-3" style="background: rgba(27, 54, 93, 0.1); border-left: 4px solid var(--kaust-blue); border-radius: 8px;">
                        <p><strong>${result.summary}</strong></p>
                    </div>
                    <small class="text-muted mt-2 d-block">
                        <i class="fas fa-info-circle"></i> 
                        Analyzed ${result.scenes || 0} scenes • 
                        ${result.frames_processed || 0} frames processed
                        ${result.video_duration ? ` • ${result.video_duration}` : ''}
                    </small>
                </div>
            `;
        }

        // Show sample captions
        let captionsHtml = '';
        if (result.captions && Object.keys(result.captions).length > 0) {
            const captionEntries = Object.entries(result.captions).slice(0, 5); // Show first 5
            captionsHtml = captionEntries.map(([frame, caption]) =>
                `<div class="caption-item mb-2 p-2" style="background: rgba(0, 167, 157, 0.05); border-radius: 6px;">
                    <small class="text-muted"><i class="fas fa-play-circle"></i> Frame ${frame}:</small><br>
                    ${caption}
                </div>`
            ).join('');

            content += `
                <div class="caption-result">
                    <h6><i class="fas fa-list"></i> Sample Captions</h6>
                    ${captionsHtml}
                    ${Object.keys(result.captions).length > 5 ? 
                        `<small class="text-muted">... and ${Object.keys(result.captions).length - 5} more captions</small>` 
                        : ''}
                </div>
            `;
        }

        // Fallback if no content
        if (!content) {
            content = `
                <div class="no-results">
                    <i class="fas fa-exclamation-triangle text-warning"></i>
                    <p>Video was processed but no results were generated. This might be due to:</p>
                    <ul class="text-start">
                        <li>Video format not supported</li>
                        <li>Video too short or no clear scenes</li>
                        <li>Processing error occurred</li>
                    </ul>
                </div>
            `;
        }

        resultsContent.innerHTML = content;
    }

    // Utility Functions
    resetUpload() {
        const preview = document.getElementById('image-preview');
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');
        if (preview) preview.style.display = 'none';
        if (uploadZone) uploadZone.style.display = 'block';
        if (fileInput) fileInput.value = '';
        this.clearResults();
    }

    clearResults() {
        const resultsContent = document.getElementById('results-content');
        if (!resultsContent) return;
        resultsContent.innerHTML = `
            <div class="no-results">
                <i class="fas fa-lightbulb"></i>
                <p>Upload an image, start camera, or analyze video to see AI-generated captions and insights.</p>
            </div>
        `;
        this.liveSummaries = []; // Clear live summaries
    }

    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loading-overlay');
        const text = document.getElementById('loading-text');
        if (text) text.textContent = message;
        if (overlay) overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.style.display = 'none';
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        if (!toast) return;

        const toastBody = toast.querySelector('.toast-body');
        const toastHeader = toast.querySelector('.toast-header');

        // Set icon based on type
        const iconMap = {
            'success': 'fas fa-check-circle text-success',
            'error': 'fas fa-exclamation-circle text-danger',
            'warning': 'fas fa-exclamation-triangle text-warning',
            'info': 'fas fa-info-circle text-info'
        };

        const icon = toastHeader.querySelector('i');
        if (icon) icon.className = iconMap[type] || iconMap['info'];

        if (toastBody) toastBody.textContent = message;

        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing KAUST Vision Captioning System...');
    window.app = new VisionCaptioningApp();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.app && window.app.websocket) {
        try { window.app.websocket.close(); } catch {}
    }
});