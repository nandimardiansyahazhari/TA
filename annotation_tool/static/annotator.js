/**
 * SmartAnnotator Frontend Engine
 * Controls state, canvas rendering, user interactions, and backend API integration.
 */

// Application State
const state = {
    classes: {},            // Class Map (ID -> Name)
    classColors: [
        '#3b82f6', // Neon Blue
        '#f97316', // Neon Orange
        '#10b981', // Neon Green
        '#8b5cf6', // Neon Purple
        '#ec4899', // Neon Pink
        '#eab308'  // Neon Yellow
    ],
    currentSubpath: '',
    images: [],             // Image list in current set
    currentImageIndex: -1,
    imageFilter: 'all',     // all, annotated, unannotated
    searchQuery: '',

    // Canvas transform states
    zoom: 1.0,
    panX: 0,
    panY: 0,
    isPanning: false,
    panStartX: 0,
    panStartY: 0,

    // Touch gesture states
    isDoubleTouching: false,
    touchStartDist: 0,
    touchStartMid: { x: 0, y: 0 },
    panStartMidX: 0,
    panStartMidY: 0,
    zoomStartVal: 1.0,

    // Annotation States
    boxes: [],              // Box list: {class_id, x_center, y_center, width, height}
    selectedBoxIndex: -1,
    activeClassId: 0,
    toolMode: 'select',     // select, draw
    isDrawing: false,
    drawStart: { x: 0, y: 0 },
    drawEnd: { x: 0, y: 0 },

    // Resize/Drag states
    dragMode: null,         // 'move', 'nw', 'ne', 'se', 'sw', 'n', 'e', 's', 'w'
    dragStartBox: null,     // copy of box when drag started
    dragStartMouse: { x: 0, y: 0 },

    // Image element
    img: new Image(),
    imgLoaded: false,

    // Save status
    hasUnsavedChanges: false,

    // Custom hotkey class mappings
    classShortcuts: {},

    // Custom class colors mapping
    classColorsMap: {}
};

// Get color for class (custom if defined, otherwise neon fallback)
function getClassColor(classId) {
    if (state.classColorsMap && state.classColorsMap[classId]) {
        return state.classColorsMap[classId];
    }
    return state.classColors[classId % state.classColors.length];
}

// Handle Sizes
const HANDLE_SIZE = 7;

// DOM Elements
const elements = {
    imageList: document.getElementById('image-list'),
    imageSearch: document.getElementById('image-search'),
    imageCount: document.getElementById('image-count'),
    countAll: document.getElementById('count-all'),
    countUnannotated: document.getElementById('count-unannotated'),
    countAnnotated: document.getElementById('count-annotated'),
    filterBtns: document.querySelectorAll('.filter-btn'),
    canvasPanel: document.getElementById('canvas-panel'),
    canvasContainer: document.getElementById('canvas-container'),
    canvasViewport: document.getElementById('canvas-viewport'),
    canvas: document.getElementById('annotation-canvas'),
    currentImageName: document.getElementById('current-image-name'),
    imageResolution: document.getElementById('image-resolution'),
    zoomLevel: document.getElementById('zoom-level'),
    modeIcon: document.getElementById('mode-icon'),
    modeText: document.getElementById('mode-text'),
    classList: document.getElementById('class-list'),
    toolSelect: document.getElementById('tool-select'),
    toolDraw: document.getElementById('tool-draw'),
    btnSave: document.getElementById('btn-save'),
    btnAutoAnnotate: document.getElementById('btn-auto-annotate'),
    btnAutoAnnotateFolder: document.getElementById('btn-auto-annotate-folder'),
    btnClearBoxes: document.getElementById('btn-clear-boxes'),
    saveStatus: document.getElementById('save-status'),
    uploadDropzone: document.getElementById('upload-dropzone'),
    fileInput: document.getElementById('file-input'),
    folderInput: document.getElementById('folder-input'),
    btnBrowseFile: document.getElementById('btn-browse-file'),
    btnBrowseFolder: document.getElementById('btn-browse-folder'),
    projectSelector: document.getElementById('project-selector'),
    btnNewProject: document.getElementById('btn-new-project'),
    projectModal: document.getElementById('project-modal'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    btnCancelProject: document.getElementById('btn-cancel-project'),
    projectForm: document.getElementById('project-form'),
    projShortcuts: document.getElementById('proj-shortcuts'),
    btnProjectSettings: document.getElementById('btn-project-settings'),
    projectSettingsModal: document.getElementById('project-settings-modal'),
    btnCloseSettingsModal: document.getElementById('btn-close-settings-modal'),
    btnCancelSettings: document.getElementById('btn-cancel-settings'),
    projectSettingsForm: document.getElementById('project-settings-form'),
    settingsProjName: document.getElementById('settings-proj-name'),
    settingsClassesList: document.getElementById('settings-classes-list'),
    btnTrainYolo: document.getElementById('btn-train-yolo'),
    trainingModal: document.getElementById('training-modal'),
    btnCloseTrainingModal: document.getElementById('btn-close-training-modal'),
    btnCancelTraining: document.getElementById('btn-cancel-training'),
    trainingForm: document.getElementById('training-form'),
    trainEpochs: document.getElementById('train-epochs'),
    trainBatch: document.getElementById('train-batch'),
    trainImgsz: document.getElementById('train-imgsz'),
    trainingLogConsole: document.getElementById('training-log-console'),
    trainingStatusText: document.getElementById('training-status-text'),
    btnSubmitTrain: document.getElementById('btn-submit-train'),
    
    // Analytics Dashboard Elements
    btnAnalytics: document.getElementById('btn-analytics'),
    analyticsModal: document.getElementById('analytics-modal'),
    btnCloseAnalyticsModal: document.getElementById('btn-close-analytics-modal'),
    btnCloseAnalytics: document.getElementById('btn-close-analytics'),
    statTotalImages: document.getElementById('stat-total-images'),
    statAnnotatedImages: document.getElementById('stat-annotated-images'),
    statUnannotatedImages: document.getElementById('stat-unannotated-images'),
    statAvgBoxes: document.getElementById('stat-avg-boxes'),
    statProgressPercent: document.getElementById('stat-progress-percent'),
    statProgressBar: document.getElementById('stat-progress-bar'),
    analyticsClassChart: document.getElementById('analytics-class-chart'),

    // Video Processing Elements
    btnBrowseVideo: document.getElementById('btn-browse-video'),
    videoModal: document.getElementById('video-modal'),
    btnCloseVideoModal: document.getElementById('btn-close-video-modal'),
    btnCancelVideo: document.getElementById('btn-cancel-video'),
    videoForm: document.getElementById('video-form'),
    videoFileInput: document.getElementById('video-file-input'),
    videoFrameStep: document.getElementById('video-frame-step'),
    videoAutoAnnotate: document.getElementById('video-auto-annotate'),
    videoProgressContainer: document.getElementById('video-progress-container'),
    videoProgressStatus: document.getElementById('video-progress-status'),
    videoProgressPercent: document.getElementById('video-progress-percent'),
    videoProgressBar: document.getElementById('video-progress-bar'),
    btnSubmitVideo: document.getElementById('btn-submit-video')
};

// Canvas context
const ctx = elements.canvas.getContext('2d');

// Initialize application
async function init() {
    setupEventListeners();
    await fetchProjects();
    await fetchConfig();
    await fetchTree();
}

// Fetch all available projects from the backend
async function fetchProjects() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();
        state.projects = data.projects;
        state.activeProjectId = data.active_id;

        // Populate dropdown selector
        elements.projectSelector.innerHTML = '';
        state.projects.forEach(proj => {
            const opt = document.createElement('option');
            opt.value = proj.id;
            opt.textContent = proj.name;
            opt.selected = (proj.id === state.activeProjectId);
            elements.projectSelector.appendChild(opt);
        });

        // Update header badge indicators
        const activeProj = state.projects.find(p => p.id === state.activeProjectId);
        if (activeProj) {
            const activeProjectBadge = document.getElementById('active-project-badge');
            const activeFormatBadge = document.getElementById('active-format-badge');
            if (activeProjectBadge) {
                activeProjectBadge.textContent = activeProj.name;
            }
            if (activeFormatBadge) {
                let fmtStr = "YOLO";
                if (activeProj.format === "voc") fmtStr = "Pascal VOC";
                else if (activeProj.format === "json") fmtStr = "JSON";
                activeFormatBadge.textContent = "Format: " + fmtStr;
            }
        }
    } catch (err) {
        console.error("Error fetching projects list", err);
    }
}

// Switches project on backend and resets application state
async function switchProject(projectId) {
    if (state.hasUnsavedChanges) {
        const confirmSwitch = confirm("You have unsaved annotations. Switch project anyway? (Unsaved changes will be lost.)");
        if (!confirmSwitch) {
            elements.projectSelector.value = state.activeProjectId;
            return;
        }
    }

    showStatus("Switching project...", "saving");
    try {
        const response = await fetch('/api/projects/active', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_id: projectId })
        });
        const data = await response.json();
        if (data.success) {
            // Reset state parameters
            state.currentSubpath = '';
            state.images = [];
            state.currentImageIndex = -1;
            state.boxes = [];
            state.selectedBoxIndex = -1;
            state.imgLoaded = false;
            state.hasUnsavedChanges = false;

            // Reload configs, directories and badges
            await fetchProjects();
            await fetchConfig();
            await fetchTree();

            // Clear canvas and header labels
            ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
            elements.currentImageName.textContent = "Select an image...";
            elements.imageResolution.textContent = "0x0";
            elements.imageList.innerHTML = '';
            elements.imageCount.textContent = '0';

            showStatus("Project loaded successfully.", "idle");
        } else {
            alert("Failed to switch project: " + (data.error || "Unknown error"));
            elements.projectSelector.value = state.activeProjectId;
        }
    } catch (err) {
        console.error("Switch project error", err);
        showStatus("Switch project failed", "unsaved");
        elements.projectSelector.value = state.activeProjectId;
    }
}

function openProjectModal() {
    elements.projectModal.classList.remove('hidden');
}

function closeProjectModal() {
    elements.projectModal.classList.add('hidden');
}

async function createProject(e) {
    e.preventDefault();

    const name = document.getElementById('proj-name').value;
    const imagesDir = document.getElementById('proj-images-dir').value;
    const labelsDir = document.getElementById('proj-labels-dir').value;
    const classes = document.getElementById('proj-classes').value.split(',').map(c => c.trim()).filter(c => c);
    const format = document.getElementById('proj-format').value;
    const shortcuts = elements.projShortcuts.value.split(',').map(s => s.trim());

    showStatus("Creating project...", "saving");

    try {
        const response = await fetch('/api/projects/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                images_dir: imagesDir,
                labels_dir: labelsDir,
                classes,
                format,
                class_shortcuts: shortcuts
            })
        });
        const data = await response.json();

        if (data.success) {
            closeProjectModal();
            document.getElementById('project-form').reset();

            // Reset state parameters
            state.currentSubpath = '';
            state.images = [];
            state.currentImageIndex = -1;
            state.boxes = [];
            state.selectedBoxIndex = -1;
            state.imgLoaded = false;
            state.hasUnsavedChanges = false;

            // Reload configs and directories
            await fetchProjects();
            await fetchConfig();
            await fetchTree();

            // Clear canvas and headers
            ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
            elements.currentImageName.textContent = "Select an image...";
            elements.imageResolution.textContent = "0x0";
            elements.imageList.innerHTML = '';
            elements.imageCount.textContent = '0';

            showStatus("Project created successfully.", "idle");
        } else {
            alert("Failed to create project: " + (data.error || "Unknown error"));
            showStatus("Project creation failed", "unsaved");
        }
    } catch (err) {
        console.error("Create project error", err);
        showStatus("Create project failed", "unsaved");
    }
}

function openSettingsModal() {
    const activeProj = state.projects.find(p => p.id === state.activeProjectId);
    if (!activeProj) {
        alert("No active project found.");
        return;
    }

    elements.settingsProjName.value = activeProj.name;
    elements.settingsClassesList.innerHTML = '';

    Object.keys(state.classes).forEach((idStr) => {
        const id = parseInt(idStr);
        const className = state.classes[id];
        const currentShortcut = (state.classShortcuts && state.classShortcuts[id]) !== undefined
            ? state.classShortcuts[id]
            : id;
        const currentColor = getClassColor(id);

        const row = document.createElement('div');
        row.className = 'class-setting-row';
        row.setAttribute('data-class-id', id);
        row.style.cssText = 'display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);';

        row.innerHTML = `
            <div class="class-setting-info" style="display: flex; align-items: center; gap: 8px; flex: 1;">
                <span class="class-color-indicator" style="background-color: ${currentColor}; width: 12px; height: 12px; border-radius: 50%; display: inline-block;"></span>
                <span style="font-weight: 500; font-size: 0.85rem; color: var(--text-primary);">${className}</span>
            </div>
            
            <div class="class-setting-controls" style="display: flex; align-items: center; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <label style="font-size: 0.75rem; color: var(--text-secondary); margin: 0;">Key:</label>
                    <input type="text" class="class-shortcut-input" value="${currentShortcut}" maxlength="1" style="width: 40px; text-align: center; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 4px; padding: 2px 4px; font-size: 0.8rem; font-weight: 600;">
                </div>
                
                <div style="display: flex; align-items: center; gap: 6px;">
                    <label style="font-size: 0.75rem; color: var(--text-secondary); margin: 0;">Color:</label>
                    <input type="color" class="class-color-input" value="${currentColor}" style="width: 32px; height: 24px; border: none; padding: 0; background: transparent; cursor: pointer; border-radius: 4px;">
                </div>
            </div>
        `;

        const colorInput = row.querySelector('.class-color-input');
        const indicator = row.querySelector('.class-color-indicator');
        colorInput.addEventListener('input', (e) => {
            indicator.style.backgroundColor = e.target.value;
        });

        elements.settingsClassesList.appendChild(row);
    });

    elements.projectSettingsModal.classList.remove('hidden');
}

function closeSettingsModal() {
    elements.projectSettingsModal.classList.add('hidden');
}

async function saveProjectSettings(e) {
    e.preventDefault();

    showStatus("Saving project settings...", "saving");

    const shortcuts = {};
    const colors = {};

    const rows = elements.settingsClassesList.querySelectorAll('.class-setting-row');
    rows.forEach(row => {
        const classId = row.getAttribute('data-class-id');
        const shortcutVal = row.querySelector('.class-shortcut-input').value.trim();
        const colorVal = row.querySelector('.class-color-input').value;

        if (shortcutVal) {
            shortcuts[classId] = shortcutVal;
        }
        colors[classId] = colorVal;
    });

    try {
        const response = await fetch('/api/projects/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.activeProjectId,
                class_shortcuts: shortcuts,
                class_colors: colors
            })
        });
        const data = await response.json();

        if (data.success) {
            closeSettingsModal();
            showStatus("Settings updated successfully.", "idle");

            await fetchProjects();
            await fetchConfig();
            redraw();
        } else {
            alert("Failed to save settings: " + (data.error || "Unknown error"));
            showStatus("Settings update failed", "unsaved");
        }
    } catch (err) {
        console.error("Save settings error", err);
        showStatus("Settings update failed", "unsaved");
    }
}

let trainingPollInterval = null;

function openTrainingModal() {
    elements.trainingModal.classList.remove('hidden');
    elements.trainingLogConsole.textContent = "Waiting to start training...\nClick 'Start Training' to begin.";
    updateTrainingUI("idle");
    checkTrainingStatus();
}

function closeTrainingModal() {
    elements.trainingModal.classList.add('hidden');
    stopTrainingPoll();
}

function updateTrainingUI(status, device = "Unknown") {
    if (status === "training") {
        elements.trainingStatusText.textContent = `Status: Training... [Device: ${device}]`;
        elements.trainingStatusText.style.color = "#3b82f6";
        elements.btnSubmitTrain.innerHTML = "<i class='fa-solid fa-stop'></i> Stop Training";
        elements.btnSubmitTrain.classList.remove('btn-accent');
        elements.btnSubmitTrain.classList.add('btn-danger');

        elements.trainEpochs.disabled = true;
        elements.trainBatch.disabled = true;
        elements.trainImgsz.disabled = true;
    } else {
        const text = status.charAt(0).toUpperCase() + status.slice(1);
        elements.trainingStatusText.textContent = `Status: ${text}`;

        if (status === "completed") {
            elements.trainingStatusText.style.color = "#10b981";
        } else if (status === "error") {
            elements.trainingStatusText.style.color = "#ef4444";
        } else {
            elements.trainingStatusText.style.color = "var(--text-secondary)";
        }

        elements.btnSubmitTrain.innerHTML = "<i class='fa-solid fa-play'></i> Start Training";
        elements.btnSubmitTrain.classList.remove('btn-danger');
        elements.btnSubmitTrain.classList.add('btn-accent');

        elements.trainEpochs.disabled = false;
        elements.trainBatch.disabled = false;
        elements.trainImgsz.disabled = false;
    }
}

async function handleTrainingSubmit(e) {
    e.preventDefault();

    if (elements.btnSubmitTrain.classList.contains('btn-danger')) {
        if (!confirm("Are you sure you want to stop the training process?")) {
            return;
        }
        try {
            const res = await fetch('/api/train/stop', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                updateTrainingUI("idle");
            }
        } catch (err) {
            console.error("Failed to stop training", err);
        }
        return;
    }

    const epochs = parseInt(elements.trainEpochs.value);
    const batch = parseInt(elements.trainBatch.value);
    const imgsz = parseInt(elements.trainImgsz.value);

    elements.trainingLogConsole.textContent = "Initiating training subprocess...\n";
    updateTrainingUI("training");

    try {
        const res = await fetch('/api/train/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs, batch, imgsz })
        });
        const data = await res.json();

        if (data.success) {
            startTrainingPoll();
        } else {
            elements.trainingLogConsole.textContent += `Failed to start training: ${data.error}\n`;
            updateTrainingUI("error");
        }
    } catch (err) {
        console.error("Start training error", err);
        elements.trainingLogConsole.textContent += `Connection error: ${err.message}\n`;
        updateTrainingUI("error");
    }
}

async function checkTrainingStatus() {
    try {
        const res = await fetch('/api/train/status');
        const data = await res.json();

        if (data.logs && data.logs.length > 0) {
            elements.trainingLogConsole.textContent = data.logs.join('\n');
            elements.trainingLogConsole.scrollTop = elements.trainingLogConsole.scrollHeight;
        }

        updateTrainingUI(data.status, data.device || "Unknown");

        if (data.status !== "training") {
            stopTrainingPoll();
        } else {
            startTrainingPoll();
        }
    } catch (err) {
        console.error("Check training status error", err);
    }
}

function startTrainingPoll() {
    if (!trainingPollInterval) {
        trainingPollInterval = setInterval(checkTrainingStatus, 1500);
    }
}

function stopTrainingPoll() {
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
        trainingPollInterval = null;
    }
}

// Analytics Dashboard Modal controls
async function openAnalyticsModal() {
    elements.analyticsModal.classList.remove('hidden');
    
    // Reset values to loading states
    elements.statTotalImages.textContent = '...';
    elements.statAnnotatedImages.textContent = '...';
    elements.statUnannotatedImages.textContent = '...';
    elements.statAvgBoxes.textContent = '...';
    elements.statProgressPercent.textContent = '0%';
    elements.statProgressBar.style.width = '0%';
    elements.analyticsClassChart.innerHTML = '<div style="color: var(--text-secondary); font-size: 0.8rem; text-align: center; padding: 12px;">Loading charts...</div>';

    try {
        const response = await fetch('/api/analytics');
        const data = await response.json();
        
        elements.statTotalImages.textContent = data.total_images;
        elements.statAnnotatedImages.textContent = data.annotated_images;
        elements.statUnannotatedImages.textContent = data.unannotated_images;
        elements.statAvgBoxes.textContent = data.avg_boxes_per_image;
        
        const progress = data.total_images > 0 ? Math.round((data.annotated_images / data.total_images) * 100) : 0;
        elements.statProgressPercent.textContent = `${progress}%`;
        elements.statProgressBar.style.width = `${progress}%`;
        
        // Populate chart
        elements.analyticsClassChart.innerHTML = '';
        const counts = Object.values(data.class_counts);
        const maxCount = Math.max(...counts, 1);
        
        Object.keys(data.class_counts).forEach(className => {
            const count = data.class_counts[className];
            const percent = (count / maxCount) * 100;
            
            // Map className to classId to retrieve custom color
            const classId = Object.keys(state.classes).find(key => state.classes[key] === className);
            const barColor = getClassColor(classId !== undefined ? parseInt(classId) : 0);
            
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; align-items: center; justify-content: space-between; font-size: 0.85rem;';
            row.innerHTML = `
                <span style="width: 100px; color: var(--text-secondary); text-overflow: ellipsis; overflow: hidden; white-space: nowrap; font-weight: 500;">${className}</span>
                <div style="flex: 1; margin: 0 12px; background: rgba(255,255,255,0.06); height: 16px; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.03);">
                    <div style="background: ${barColor}; width: ${percent}%; height: 100%; border-radius: 8px; transition: width 0.6s ease;"></div>
                </div>
                <span style="width: 40px; text-align: right; font-weight: 600; color: #fff;">${count}</span>
            `;
            elements.analyticsClassChart.appendChild(row);
        });
        
        if (counts.length === 0) {
            elements.analyticsClassChart.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 12px;">No objects detected in labels yet.</div>';
        }
        
    } catch (err) {
        console.error("Failed to load analytics data", err);
        elements.analyticsClassChart.innerHTML = '<div style="color: var(--color-danger); font-size: 0.8rem; text-align: center; padding: 12px;">Error loading data.</div>';
    }
}

function closeAnalyticsModal() {
    elements.analyticsModal.classList.add('hidden');
}

// Video Processing Modal controls
function openVideoModal() {
    elements.videoModal.classList.remove('hidden');
    elements.videoForm.reset();
    elements.videoProgressContainer.classList.add('hidden');
    elements.videoProgressBar.style.width = '0%';
    elements.videoProgressPercent.textContent = '0%';
    elements.btnSubmitVideo.disabled = false;
    elements.btnCancelVideo.disabled = false;
}

function closeVideoModal() {
    elements.videoModal.classList.add('hidden');
}

async function handleVideoSubmit(e) {
    e.preventDefault();
    
    const file = elements.videoFileInput.files[0];
    const frameStep = parseInt(elements.videoFrameStep.value);
    const autoAnnotate = elements.videoAutoAnnotate.checked;
    
    if (!file) {
        alert("Please select a video file first.");
        return;
    }
    
    elements.videoProgressContainer.classList.remove('hidden');
    elements.videoProgressStatus.textContent = "Uploading video...";
    elements.videoProgressPercent.textContent = "0%";
    elements.videoProgressBar.style.width = "0%";
    elements.btnSubmitVideo.disabled = true;
    elements.btnCancelVideo.disabled = true;
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("frame_step", frameStep);
    formData.append("auto_annotate", autoAnnotate);
    
    try {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `/api/video/process/${state.currentSubpath}`);
        
        xhr.upload.addEventListener("progress", (event) => {
            if (event.lengthComputable) {
                const percent = Math.round((event.loaded / event.total) * 90);
                elements.videoProgressPercent.textContent = `${percent}%`;
                elements.videoProgressBar.style.width = `${percent}%`;
                if (percent >= 90) {
                    elements.videoProgressStatus.textContent = "Processing and extracting frames (please wait)...";
                }
            }
        });
        
        xhr.addEventListener("load", async () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                elements.videoProgressPercent.textContent = "100%";
                elements.videoProgressBar.style.width = "100%";
                elements.videoProgressStatus.textContent = `Success: Extracted ${data.count} frames!`;
                
                setTimeout(async () => {
                    closeVideoModal();
                    showStatus("Video processed successfully.", "idle");
                    await fetchTree();
                    await fetchImages();
                }, 1500);
            } else {
                let errText = "Unknown error occurred.";
                try {
                    const data = JSON.parse(xhr.responseText);
                    errText = data.error || errText;
                } catch(e) {}
                
                elements.videoProgressStatus.textContent = `Error: ${errText}`;
                elements.videoProgressStatus.style.color = "var(--color-danger)";
                elements.btnSubmitVideo.disabled = false;
                elements.btnCancelVideo.disabled = false;
            }
        });
        
        xhr.addEventListener("error", () => {
            elements.videoProgressStatus.textContent = "Error: Network connection failed.";
            elements.videoProgressStatus.style.color = "var(--color-danger)";
            elements.btnSubmitVideo.disabled = false;
            elements.btnCancelVideo.disabled = false;
        });
        
        xhr.send(formData);
        
    } catch (err) {
        console.error("Video upload/process error", err);
        elements.videoProgressStatus.textContent = `Error: ${err.message}`;
        elements.btnSubmitVideo.disabled = false;
        elements.btnCancelVideo.disabled = false;
    }
}

// Fetch class labels
async function fetchConfig() {
    try {
        const response = await fetch('/api/config');
        const data = await response.json();
        state.classes = data.classes;
        state.classShortcuts = data.class_shortcuts || {};
        state.classColorsMap = data.class_colors || {};
        renderClassList();
    } catch (err) {
        console.error("Failed to load configuration", err);
        showStatus("Error loading classes config", "unsaved");
    }
}

// Fetch folder directory tree
async function fetchTree() {
    try {
        const response = await fetch('/api/tree');
        const data = await response.json();
        renderDirectoryTree(data);

        // Auto select first subpath
        if (!state.currentSubpath && data.children && data.children.length > 0) {
            selectSubpath(data.children[0].relative_path);
        }
    } catch (err) {
        console.error("Failed to fetch directory tree", err);
    }
}

// Render the local directory tree sidebar panel
function renderDirectoryTree(treeData) {
    const container = document.getElementById('directory-tree');
    if (!container) return;
    container.innerHTML = '';

    if (treeData.children) {
        treeData.children.forEach(child => {
            container.appendChild(createTreeNode(child));
        });
    }
}

// Recursively builds HTML tree node elements
function createTreeNode(nodeData) {
    const nodeEl = document.createElement('div');
    nodeEl.className = 'tree-node';

    const headerEl = document.createElement('div');
    headerEl.className = 'tree-node-header';
    if (state.currentSubpath === nodeData.relative_path) {
        headerEl.classList.add('active');
    }
    headerEl.setAttribute('data-path', nodeData.relative_path);

    const hasChildren = nodeData.children && nodeData.children.length > 0;

    const toggleEl = document.createElement('span');
    toggleEl.className = 'tree-folder-toggle';
    if (hasChildren) {
        toggleEl.innerHTML = '<i class="fa-solid fa-chevron-down"></i>';
    }

    const iconEl = document.createElement('span');
    iconEl.className = 'tree-folder-icon';
    iconEl.innerHTML = '<i class="fa-solid fa-folder"></i>';

    const nameEl = document.createElement('span');
    nameEl.className = 'tree-node-name';
    nameEl.textContent = nodeData.name;

    headerEl.appendChild(toggleEl);
    headerEl.appendChild(iconEl);
    headerEl.appendChild(nameEl);

    // Add folder delete button if this is not the root folder
    if (nodeData.relative_path) {
        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'tree-folder-delete';
        deleteBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
        deleteBtn.style.marginLeft = 'auto';
        deleteBtn.style.padding = '2px 6px';
        deleteBtn.style.borderRadius = '4px';
        deleteBtn.style.color = 'var(--text-muted)';
        deleteBtn.style.cursor = 'pointer';
        deleteBtn.style.opacity = '0';
        deleteBtn.style.transition = 'opacity 0.15s, color 0.15s, background-color 0.15s';

        headerEl.addEventListener('mouseenter', () => { deleteBtn.style.opacity = '1'; });
        headerEl.addEventListener('mouseleave', () => { deleteBtn.style.opacity = '0'; });

        deleteBtn.addEventListener('mouseenter', () => {
            deleteBtn.style.color = 'var(--color-danger)';
            deleteBtn.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
        });
        deleteBtn.addEventListener('mouseleave', () => {
            deleteBtn.style.color = 'var(--text-muted)';
            deleteBtn.style.backgroundColor = 'transparent';
        });

        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation(); // Prevent folder activation selection
            const confirmDelete = confirm(`Are you sure you want to delete the folder "${nodeData.name}"? This will permanently delete all images and annotation labels inside it.`);
            if (!confirmDelete) return;

            showStatus("Deleting folder...", "saving");
            try {
                const response = await fetch(`/api/delete-folder/${nodeData.relative_path}`, {
                    method: 'DELETE'
                });
                const resData = await response.json();
                if (resData.success) {
                    showStatus("Folder deleted successfully.", "idle");
                    // If we deleted the current active subpath, reset state
                    if (state.currentSubpath && (state.currentSubpath === nodeData.relative_path || state.currentSubpath.startsWith(nodeData.relative_path + '/'))) {
                        state.currentSubpath = '';
                        state.images = [];
                        state.currentImageIndex = -1;
                        state.boxes = [];
                        clearWorkspace();
                    }
                    await fetchTree();
                } else {
                    alert("Failed to delete folder: " + (resData.error || "Unknown error"));
                    showStatus("Failed to delete folder", "unsaved");
                }
            } catch (err) {
                console.error("Delete folder error", err);
                alert("An error occurred while deleting the folder.");
                showStatus("Error deleting folder", "unsaved");
            }
        });

        headerEl.appendChild(deleteBtn);
    }

    nodeEl.appendChild(headerEl);

    const childrenEl = document.createElement('div');
    childrenEl.className = 'tree-node-children';

    if (hasChildren) {
        nodeData.children.forEach(child => {
            childrenEl.appendChild(createTreeNode(child));
        });
        nodeEl.appendChild(childrenEl);

        toggleEl.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleEl.classList.toggle('collapsed');
            childrenEl.classList.toggle('hidden');
        });
    }

    headerEl.addEventListener('click', async () => {
        if (state.currentSubpath !== nodeData.relative_path) {
            if (state.hasUnsavedChanges) {
                await saveAnnotations();
            }
            selectSubpath(nodeData.relative_path);
        }
    });

    return nodeEl;
}

function selectSubpath(subpath) {
    state.currentSubpath = subpath;

    document.querySelectorAll('.tree-node-header').forEach(header => {
        if (header.getAttribute('data-path') === subpath) {
            header.classList.add('active');
        } else {
            header.classList.remove('active');
        }
    });

    fetchImages();
}

// Fetch images for the current active subpath folder
async function fetchImages(preserveActiveImage = false) {
    try {
        const response = await fetch(`/api/images/${state.currentSubpath}`);
        const data = await response.json();
        state.images = data.images;

        renderGalleryList();

        if (state.images.length > 0) {
            if (!preserveActiveImage) {
                loadImageIndex(0);
            } else {
                updateGalleryItemStatuses();
            }
        } else {
            clearWorkspace();
        }
    } catch (err) {
        console.error("Failed to fetch images", err);
    }
}

// Render the right sidebar class labels list
function renderClassList() {
    elements.classList.innerHTML = '';
    Object.keys(state.classes).forEach((idStr) => {
        const id = parseInt(idStr);
        const name = state.classes[id];

        const row = document.createElement('div');
        row.className = `class-row ${state.activeClassId === id ? 'active' : ''}`;
        row.setAttribute('data-class-id', id);

        // Assign a color
        const color = getClassColor(id);

        // Retrieve custom shortcut if defined, otherwise default to its index ID
        const shortcutKey = (state.classShortcuts && state.classShortcuts[id]) !== undefined
            ? state.classShortcuts[id]
            : id;

        row.innerHTML = `
            <div class="class-info-left">
                <span class="class-color-indicator" style="background-color: ${color}"></span>
                <span>${name}</span>
            </div>
            <span class="class-shortcut">${shortcutKey}</span>
        `;

        row.addEventListener('click', () => {
            selectClass(id);
        });

        elements.classList.appendChild(row);
    });

    // Populate mobile class selector
    const mobileClassSelector = document.getElementById('mobile-class-selector');
    if (mobileClassSelector) {
        mobileClassSelector.innerHTML = '';
        Object.keys(state.classes).forEach((idStr) => {
            const id = parseInt(idStr);
            const name = state.classes[id];
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = name;
            opt.selected = state.activeClassId === id;
            mobileClassSelector.appendChild(opt);
        });
    }
}

function selectClass(id) {
    state.activeClassId = id;
    document.querySelectorAll('.class-row').forEach(row => {
        if (parseInt(row.getAttribute('data-class-id')) === id) {
            row.classList.add('active');
        } else {
            row.classList.remove('active');
        }
    });
    // Sync mobile select element
    const mobileClassSelector = document.getElementById('mobile-class-selector');
    if (mobileClassSelector) {
        mobileClassSelector.value = id;
    }

    // Update selected box class_id dynamically
    if (state.selectedBoxIndex !== -1 && state.toolMode === 'select') {
        state.boxes[state.selectedBoxIndex].class_id = id;
        state.hasUnsavedChanges = true;
        showStatus("Unsaved changes", "unsaved");
        redraw();
    }
}

// Helper to update filter button count badges
function updateFilterCounts() {
    let countAll = state.images.length;
    let countAnnotated = state.images.filter(img => img.status === 'annotated').length;
    let countUnannotated = state.images.filter(img => img.status === 'unannotated' || img.status === 'semi-annotated').length;

    if (elements.countAll) elements.countAll.textContent = countAll;
    if (elements.countAnnotated) elements.countAnnotated.textContent = countAnnotated;
    if (elements.countUnannotated) elements.countUnannotated.textContent = countUnannotated;
}

// Render the image gallery inside the left sidebar
function renderGalleryList() {
    elements.imageList.innerHTML = '';

    // Update filter count badges
    updateFilterCounts();

    // Filter & Search
    const filteredImages = state.images.filter(img => {
        const matchesSearch = img.filename.toLowerCase().includes(state.searchQuery.toLowerCase());
        const isAnnotated = img.status === 'annotated';
        const isUnannotated = img.status === 'unannotated' || img.status === 'semi-annotated';

        const matchesFilter =
            state.imageFilter === 'all' ||
            (state.imageFilter === 'annotated' && isAnnotated) ||
            (state.imageFilter === 'unannotated' && isUnannotated);
        return matchesSearch && matchesFilter;
    });

    elements.imageCount.textContent = filteredImages.length;

    filteredImages.forEach((img) => {
        // Find index in main state array
        const mainIndex = state.images.findIndex(item => item.filename === img.filename);

        const li = document.createElement('li');
        li.className = `image-item ${state.currentImageIndex === mainIndex ? 'active' : ''}`;
        li.setAttribute('data-index', mainIndex);

        li.innerHTML = `
            <div class="image-name-wrapper">
                <i class="fa-regular fa-image"></i>
                <span class="image-name" title="${img.filename}">${img.filename}</span>
            </div>
            <div class="image-meta-pills">
                ${img.box_count > 0 ? `<span class="box-count-badge">${img.box_count}</span>` : ''}
                <span class="status-dot ${img.status}"></span>
            </div>
        `;

        li.addEventListener('click', () => {
            if (state.currentImageIndex !== mainIndex) {
                // Auto save previous annotations
                if (state.hasUnsavedChanges) {
                    saveAnnotations().then(() => loadImageIndex(mainIndex));
                } else {
                    loadImageIndex(mainIndex);
                }
            }
        });

        elements.imageList.appendChild(li);
    });
}

// Update statuses of items already in DOM (avoid full rerender to prevent losing scroll position)
function updateGalleryItemStatuses() {
    // Update filter count badges
    updateFilterCounts();

    document.querySelectorAll('.image-item').forEach(li => {
        const idx = parseInt(li.getAttribute('data-index'));
        const img = state.images[idx];
        if (img) {
            const badge = li.querySelector('.box-count-badge');
            const dot = li.querySelector('.status-dot');

            // Update box count
            if (img.box_count > 0) {
                if (badge) {
                    badge.textContent = img.box_count;
                } else {
                    // Create badge
                    const pillContainer = li.querySelector('.image-meta-pills');
                    const newBadge = document.createElement('span');
                    newBadge.className = 'box-count-badge';
                    newBadge.textContent = img.box_count;
                    pillContainer.insertBefore(newBadge, dot);
                }
            } else if (badge) {
                badge.remove();
            }

            // Update status dot class
            dot.className = `status-dot ${img.status}`;
        }
    });
}

// Load image at index
function loadImageIndex(index) {
    if (index < 0 || index >= state.images.length) return;

    state.currentImageIndex = index;
    const imgItem = state.images[index];

    // Update active highlight in gallery
    document.querySelectorAll('.image-item').forEach(li => {
        if (parseInt(li.getAttribute('data-index')) === index) {
            li.classList.add('active');
            li.scrollIntoView({ block: 'nearest' });
        } else {
            li.classList.remove('active');
        }
    });

    elements.currentImageName.innerHTML = `<i class="fa-regular fa-image"></i> ${imgItem.filename}`;
    showStatus("Loading...", "saving");

    state.imgLoaded = false;
    state.img.src = `/api/image/${state.currentSubpath}/${imgItem.filename}`;

    state.img.onload = async () => {
        state.imgLoaded = true;
        elements.imageResolution.textContent = `${state.img.naturalWidth}x${state.img.naturalHeight}`;

        // Reset scale and center image
        resetZoomAndPan();

        // Fetch existing annotations
        await fetchAnnotations();
        showStatus("All changes saved", "idle");
        state.hasUnsavedChanges = false;
    };
}

// Clear drawing area if no images exist
function clearWorkspace() {
    state.currentImageIndex = -1;
    state.boxes = [];
    state.selectedBoxIndex = -1;
    elements.currentImageName.innerHTML = `<i class="fa-regular fa-image"></i> No images found`;
    elements.imageResolution.textContent = "0x0";
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
}

// Fetch labels for current active image
async function fetchAnnotations() {
    const imgItem = state.images[state.currentImageIndex];
    try {
        const response = await fetch(`/api/annotations/${state.currentSubpath}/${imgItem.filename}`);
        const data = await response.json();
        state.boxes = data.boxes || [];
        state.selectedBoxIndex = -1;
        redraw();
    } catch (err) {
        console.error("Failed to load annotations", err);
    }
}

// Save active annotations to backend
async function saveAnnotations() {
    if (state.currentImageIndex === -1) return;

    const imgItem = state.images[state.currentImageIndex];
    showStatus("Saving...", "saving");

    try {
        const response = await fetch(`/api/annotations/${state.currentSubpath}/${imgItem.filename}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                boxes: state.boxes,
                width: state.img.naturalWidth || 640,
                height: state.img.naturalHeight || 480
            })
        });

        if (response.ok) {
            // Update local state details
            imgItem.box_count = state.boxes.length;
            imgItem.status = state.boxes.length > 0 ? 'annotated' : 'unannotated';

            updateGalleryItemStatuses();
            showStatus("All changes saved", "idle");
            state.hasUnsavedChanges = false;
        } else {
            showStatus("Failed to save!", "unsaved");
        }
    } catch (err) {
        console.error("Save error", err);
        showStatus("Connection error!", "unsaved");
    }
}

// Run auto label API
async function triggerAutoAnnotation() {
    if (state.currentImageIndex === -1) return;

    const imgItem = state.images[state.currentImageIndex];
    showStatus("AI Annotating...", "saving");

    try {
        const response = await fetch(`/api/auto-annotate/${state.currentSubpath}/${imgItem.filename}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.error) {
            showStatus("AI failed: " + data.error, "unsaved");
            return;
        }

        if (data.boxes) {
            // Append or overwrite? Let's overwrite / combine.
            // Overwriting is standard for a clean review. Let's merge if some boxes exist,
            // or overwrite if empty. Let's overwrite so they see exactly what the model predicts.
            state.boxes = data.boxes;
            state.selectedBoxIndex = -1;
            state.hasUnsavedChanges = true;
            redraw();
            showStatus("Auto-labeled! Please review and save.", "unsaved");
        }
    } catch (err) {
        console.error("Auto label error", err);
        showStatus("AI service connection failed", "unsaved");
    }
}

// Run folder-wide auto annotation API
async function triggerAutoAnnotateFolder() {
    if (!state.currentSubpath) return;

    if (state.hasUnsavedChanges) {
        showStatus("Saving current annotations before running...", "saving");
        await saveAnnotations();
    }

    if (!confirm("Are you sure you want to run AI auto-labeling on all unannotated images in this folder? This will automatically detect boxes and save them directly to disk. (Already reviewed/annotated files will be skipped.)")) {
        return;
    }

    showStatus("AI Auto-Annotating Folder...", "saving");

    try {
        const response = await fetch(`/api/auto-annotate-folder/${state.currentSubpath}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.error) {
            showStatus("Folder AI failed: " + data.error, "unsaved");
            alert("Folder AI failed: " + data.error);
            return;
        }

        if (data.success) {
            showStatus(data.message, "idle");
            alert(data.message);

            // Reload image list in the active folder
            await fetchImages(true);

            // Load the current image's annotations (in case the current image was auto-annotated in the batch!)
            if (state.currentImageIndex !== -1) {
                await fetchAnnotations();
            }
        }
    } catch (err) {
        console.error("Folder auto-annotate error", err);
        showStatus("AI service connection failed", "unsaved");
    }
}

// Reset Zoom and pan, centering the image in the panel
function resetZoomAndPan() {
    if (!state.imgLoaded) return;

    const containerW = elements.canvasPanel.clientWidth;
    const containerH = elements.canvasPanel.clientHeight;
    const imgW = state.img.naturalWidth;
    const imgH = state.img.naturalHeight;

    // Choose zoom scale to fit image within viewport safely with some padding (e.g. 90%)
    const scaleX = (containerW * 0.9) / imgW;
    const scaleY = (containerH * 0.9) / imgH;
    state.zoom = Math.min(scaleX, scaleY, 1.5); // cap zoom at 1.5x on auto fit

    // Center alignment
    state.panX = Math.round((containerW - imgW * state.zoom) / 2);
    state.panY = Math.round((containerH - imgH * state.zoom) / 2);

    updateViewportTransform();
    redraw();
}

// Apply transform styles to canvas viewport wrapper
function updateViewportTransform() {
    elements.canvasViewport.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    elements.zoomLevel.textContent = `Zoom: ${Math.round(state.zoom * 100)}%`;
}

// Center-based zoom control (designed primarily for mobile zoom buttons)
function zoomCanvas(zoomIn) {
    if (!state.imgLoaded) return;

    const containerW = elements.canvasPanel.clientWidth;
    const containerH = elements.canvasPanel.clientHeight;

    const centerX = containerW / 2;
    const centerY = containerH / 2;

    // Canvas coordinates of center before zoom
    const canvasCenterX = (centerX - state.panX) / state.zoom;
    const canvasCenterY = (centerY - state.panY) / state.zoom;

    const zoomIntensity = 0.25;

    if (zoomIn) {
        state.zoom = Math.min(state.zoom * (1 + zoomIntensity), 20.0);
    } else {
        state.zoom = Math.max(state.zoom * (1 - zoomIntensity), 0.05);
    }

    // Recalculate panning offsets to keep the center coordinates pinned in the middle
    state.panX = Math.round(centerX - canvasCenterX * state.zoom);
    state.panY = Math.round(centerY - canvasCenterY * state.zoom);

    updateViewportTransform();
    redraw();
}

// Redraw Canvas Content
function redraw() {
    if (!state.imgLoaded) return;

    const imgW = state.img.naturalWidth;
    const imgH = state.img.naturalHeight;

    // Resize canvas element to match image physical dimensions
    elements.canvas.width = imgW;
    elements.canvas.height = imgH;

    // Draw the source image
    ctx.drawImage(state.img, 0, 0, imgW, imgH);

    // Draw existing bounding boxes
    state.boxes.forEach((box, index) => {
        drawBoundingBox(box, index === state.selectedBoxIndex);
    });

    // Draw drawing outline if in draw mode and drawing is active
    if (state.toolMode === 'draw' && state.isDrawing) {
        const color = getClassColor(state.activeClassId);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1.5 / state.zoom, 1);
        ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);

        const x = Math.min(state.drawStart.x, state.drawEnd.x);
        const y = Math.min(state.drawStart.y, state.drawEnd.y);
        const w = Math.abs(state.drawStart.x - state.drawEnd.x);
        const h = Math.abs(state.drawStart.y - state.drawEnd.y);

        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]); // Reset
    }
}

// Helper to draw a single bounding box on the canvas
function drawBoundingBox(box, isSelected) {
    const imgW = state.img.naturalWidth;
    const imgH = state.img.naturalHeight;

    // Convert YOLO normalized to absolute
    const w = box.width * imgW;
    const h = box.height * imgH;
    const x = box.x_center * imgW - w / 2;
    const y = box.y_center * imgH - h / 2;

    const color = getClassColor(box.class_id);

    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? Math.max(3 / state.zoom, 2) : Math.max(2 / state.zoom, 1.2);

    if (isSelected) {
        ctx.setLineDash([4 / state.zoom, 2 / state.zoom]);
    }

    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]); // Reset

    // Draw small text label above box
    const className = state.classes[box.class_id] || `Class ${box.class_id}`;
    const fontSize = Math.max(Math.round(11 / state.zoom), 10);
    ctx.font = `600 ${fontSize}px var(--font-sans)`;

    const textWidth = ctx.measureText(className).width;
    const textPadding = 4 / state.zoom;

    // Label background
    ctx.fillStyle = color;
    ctx.fillRect(
        x - (isSelected ? 1 / state.zoom : 0),
        y - fontSize - textPadding * 2,
        textWidth + textPadding * 2,
        fontSize + textPadding * 2
    );

    // Label Text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(className, x + textPadding, y - textPadding);

    // Draw corner & edge handles if selected (Select/Edit Mode only)
    if (isSelected && state.toolMode === 'select') {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = color;
        ctx.lineWidth = 1 / state.zoom;

        const size = HANDLE_SIZE / state.zoom;
        const halfSize = size / 2;

        const corners = [
            { x: x, y: y },              // nw
            { x: x + w, y: y },          // ne
            { x: x + w, y: y + h },      // se
            { x: x, y: y + h },          // sw
            { x: x + w / 2, y: y },      // n
            { x: x + w, y: y + h / 2 },  // e
            { x: x + w / 2, y: y + h },  // s
            { x: x, y: y + h / 2 }       // w
        ];

        corners.forEach(pt => {
            ctx.fillRect(pt.x - halfSize, pt.y - halfSize, size, size);
            ctx.strokeRect(pt.x - halfSize, pt.y - halfSize, size, size);
        });
    }
}

// Convert screen viewport client coordinates to image local canvas coordinates
function screenToCanvasCoords(clientX, clientY) {
    const rect = elements.canvas.getBoundingClientRect();

    // Transform coordinates based on bounding client rectangle (takes care of pan & zoom scale)
    const x = (clientX - rect.left) * (elements.canvas.width / rect.width);
    const y = (clientY - rect.top) * (elements.canvas.height / rect.height);

    return {
        x: Math.max(0, Math.min(x, elements.canvas.width)),
        y: Math.max(0, Math.min(y, elements.canvas.height))
    };
}

// Check which handle or box parts the mouse is currently hovering over
function getHitElement(canvasX, canvasY) {
    if (state.selectedBoxIndex !== -1 && state.toolMode === 'select') {
        const box = state.boxes[state.selectedBoxIndex];
        const imgW = state.img.naturalWidth;
        const imgH = state.img.naturalHeight;

        const w = box.width * imgW;
        const h = box.height * imgH;
        const x = box.x_center * imgW - w / 2;
        const y = box.y_center * imgH - h / 2;

        // Define hitbox padding depending on zoom scale
        const threshold = (HANDLE_SIZE + 4) / state.zoom;

        const handles = {
            nw: { x: x, y: y },
            ne: { x: x + w, y: y },
            se: { x: x + w, y: y + h },
            sw: { x: x, y: y + h },
            n: { x: x + w / 2, y: y },
            e: { x: x + w, y: y + h / 2 },
            s: { x: x + w / 2, y: y + h },
            w: { x: x, y: y + h / 2 }
        };

        for (const [mode, pt] of Object.entries(handles)) {
            if (Math.abs(canvasX - pt.x) < threshold && Math.abs(canvasY - pt.y) < threshold) {
                return { type: 'handle', mode: mode };
            }
        }
    }

    // Check inside any box (draw from last to first so top-most item is hit)
    const imgW = state.img.naturalWidth;
    const imgH = state.img.naturalHeight;
    for (let i = state.boxes.length - 1; i >= 0; i--) {
        const box = state.boxes[i];
        const w = box.width * imgW;
        const h = box.height * imgH;
        const x = box.x_center * imgW - w / 2;
        const y = box.y_center * imgH - h / 2;

        if (canvasX >= x && canvasX <= x + w && canvasY >= y && canvasY <= y + h) {
            return { type: 'box', index: i };
        }
    }

    return null;
}

// Convert start/end box coordinates back to normalized YOLO format
function saveYoloBoxCoords(boxIndex, x1, y1, x2, y2) {
    const imgW = state.img.naturalWidth;
    const imgH = state.img.naturalHeight;

    // Bounds check
    const rx1 = Math.max(0, Math.min(x1, imgW));
    const ry1 = Math.max(0, Math.min(y1, imgH));
    const rx2 = Math.max(0, Math.min(x2, imgW));
    const ry2 = Math.max(0, Math.min(y2, imgH));

    const w = Math.abs(rx1 - rx2);
    const h = Math.abs(ry1 - ry2);
    const xc = Math.min(rx1, rx2) + w / 2;
    const yc = Math.min(ry1, ry2) + h / 2;

    state.boxes[boxIndex] = {
        class_id: state.boxes[boxIndex] ? state.boxes[boxIndex].class_id : state.activeClassId,
        x_center: xc / imgW,
        y_center: yc / imgH,
        width: w / imgW,
        height: h / imgH
    };
}

// Create a new box from coordinates
function createNewBox(x1, y1, x2, y2) {
    const imgW = state.img.naturalWidth;
    const imgH = state.img.naturalHeight;

    const w = Math.abs(x1 - x2);
    const h = Math.abs(y1 - y2);

    // Ignore extremely tiny boxes (e.g. noise clicks)
    if (w < 4 || h < 4) return false;

    const xc = Math.min(x1, x2) + w / 2;
    const yc = Math.min(y1, y2) + h / 2;

    state.boxes.push({
        class_id: state.activeClassId,
        x_center: xc / imgW,
        y_center: yc / imgH,
        width: w / imgW,
        height: h / imgH
    });

    state.selectedBoxIndex = state.boxes.length - 1;
    state.hasUnsavedChanges = true;
    showStatus("Unsaved changes", "unsaved");
    return true;
}

// ==========================================================================
// Event Listeners & Interactions logic
// ==========================================================================

function setupEventListeners() {
    // Filter click handlers
    elements.filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.imageFilter = btn.getAttribute('data-filter');
            renderGalleryList();
        });
    });

    // Image search bar
    elements.imageSearch.addEventListener('input', (e) => {
        state.searchQuery = e.target.value;
        renderGalleryList();
    });

    // Tool buttons mode toggling
    elements.toolSelect.addEventListener('click', () => setToolMode('select'));
    elements.toolDraw.addEventListener('click', () => setToolMode('draw'));

    // Global Save
    elements.btnSave.addEventListener('click', saveAnnotations);

    // Clear boxes
    elements.btnClearBoxes.addEventListener('click', () => {
        if (state.boxes.length > 0) {
            if (confirm("Are you sure you want to remove all bounding boxes for this image?")) {
                state.boxes = [];
                state.selectedBoxIndex = -1;
                state.hasUnsavedChanges = true;
                showStatus("Unsaved changes", "unsaved");
                redraw();
            }
        }
    });

    // Run Auto Annotate
    elements.btnAutoAnnotate.addEventListener('click', triggerAutoAnnotation);
    if (elements.btnAutoAnnotateFolder) {
        elements.btnAutoAnnotateFolder.addEventListener('click', triggerAutoAnnotateFolder);
    }

    // Canvas interaction setup
    setupCanvasInteractions();

    // Window resize
    window.addEventListener('resize', () => {
        if (state.imgLoaded) {
            updateViewportTransform();
        }
    });

    // Keyboard Shortcuts
    document.addEventListener('keydown', handleKeyDown);

    // File/Folder upload events
    if (elements.uploadDropzone && elements.fileInput && elements.folderInput) {
        if (elements.btnBrowseFile) {
            elements.btnBrowseFile.addEventListener('click', (e) => {
                e.stopPropagation();
                elements.fileInput.click();
            });
        }

        if (elements.btnBrowseFolder) {
            elements.btnBrowseFolder.addEventListener('click', (e) => {
                e.stopPropagation();
                elements.folderInput.click();
            });
        }

        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleMultipleFilesUpload(e.target.files);
            }
        });

        elements.folderInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleMultipleFilesUpload(e.target.files);
            }
        });

        elements.uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.add('dragover');
        });

        elements.uploadDropzone.addEventListener('dragleave', () => {
            elements.uploadDropzone.classList.remove('dragover');
        });

        elements.uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleMultipleFilesUpload(e.dataTransfer.files);
            }
        });
    }

    // Project selection and creation events
    if (elements.projectSelector) {
        elements.projectSelector.addEventListener('change', (e) => {
            switchProject(e.target.value);
        });
    }
    if (elements.btnProjectSettings) {
        elements.btnProjectSettings.addEventListener('click', openSettingsModal);
    }
    if (elements.btnCancelSettings) {
        elements.btnCancelSettings.addEventListener('click', closeSettingsModal);
    }
    if (elements.btnCloseSettingsModal) {
        elements.btnCloseSettingsModal.addEventListener('click', closeSettingsModal);
    }
    if (elements.projectSettingsForm) {
        elements.projectSettingsForm.addEventListener('submit', saveProjectSettings);
    }
    if (elements.btnNewProject) {
        elements.btnNewProject.addEventListener('click', openProjectModal);
    }
    if (elements.btnCancelProject) {
        elements.btnCancelProject.addEventListener('click', closeProjectModal);
    }
    if (elements.btnCloseModal) {
        elements.btnCloseModal.addEventListener('click', closeProjectModal);
    }
    if (elements.projectForm) {
        elements.projectForm.addEventListener('submit', createProject);
    }

    // Model training events
    if (elements.btnTrainYolo) {
        elements.btnTrainYolo.addEventListener('click', openTrainingModal);
    }
    if (elements.btnCloseTrainingModal) {
        elements.btnCloseTrainingModal.addEventListener('click', closeTrainingModal);
    }
    if (elements.btnCancelTraining) {
        elements.btnCancelTraining.addEventListener('click', closeTrainingModal);
    }
    if (elements.trainingForm) {
        elements.trainingForm.addEventListener('submit', handleTrainingSubmit);
    }

    // Analytics events
    if (elements.btnAnalytics) {
        elements.btnAnalytics.addEventListener('click', openAnalyticsModal);
    }
    if (elements.btnCloseAnalyticsModal) {
        elements.btnCloseAnalyticsModal.addEventListener('click', closeAnalyticsModal);
    }
    if (elements.btnCloseAnalytics) {
        elements.btnCloseAnalytics.addEventListener('click', closeAnalyticsModal);
    }

    // Video processing events
    if (elements.btnBrowseVideo) {
        elements.btnBrowseVideo.addEventListener('click', openVideoModal);
    }
    if (elements.btnCloseVideoModal) {
        elements.btnCloseVideoModal.addEventListener('click', closeVideoModal);
    }
    if (elements.btnCancelVideo) {
        elements.btnCancelVideo.addEventListener('click', closeVideoModal);
    }
    if (elements.videoForm) {
        elements.videoForm.addEventListener('submit', handleVideoSubmit);
    }

    // Mobile specific quick control listeners
    const mobPrev = document.getElementById('mobile-btn-prev');
    const mobNext = document.getElementById('mobile-btn-next');
    const mobSelect = document.getElementById('mobile-tool-select');
    const mobDraw = document.getElementById('mobile-tool-draw');
    const mobClass = document.getElementById('mobile-class-selector');
    const mobSave = document.getElementById('mobile-btn-save');

    if (mobPrev) {
        mobPrev.addEventListener('click', () => {
            if (state.currentImageIndex > 0) {
                if (state.hasUnsavedChanges) {
                    saveAnnotations().then(() => loadImageIndex(state.currentImageIndex - 1));
                } else {
                    loadImageIndex(state.currentImageIndex - 1);
                }
            }
        });
    }

    if (mobNext) {
        mobNext.addEventListener('click', () => {
            if (state.currentImageIndex < state.images.length - 1) {
                if (state.hasUnsavedChanges) {
                    saveAnnotations().then(() => loadImageIndex(state.currentImageIndex + 1));
                } else {
                    loadImageIndex(state.currentImageIndex + 1);
                }
            }
        });
    }

    if (mobSelect) {
        mobSelect.addEventListener('click', () => setToolMode('select'));
    }

    if (mobDraw) {
        mobDraw.addEventListener('click', () => setToolMode('draw'));
    }

    const mobDelete = document.getElementById('mobile-btn-delete');
    if (mobDelete) {
        mobDelete.addEventListener('click', () => {
            if (state.selectedBoxIndex !== -1 && state.toolMode === 'select') {
                state.boxes.splice(state.selectedBoxIndex, 1);
                state.selectedBoxIndex = -1;
                state.hasUnsavedChanges = true;
                showStatus("Unsaved changes", "unsaved");
                redraw();
            } else {
                alert("Please select a bounding box to delete first.");
            }
        });
    }

    if (mobClass) {
        mobClass.addEventListener('change', (e) => {
            selectClass(parseInt(e.target.value));
        });
    }

    if (mobSave) {
        mobSave.addEventListener('click', saveAnnotations);
    }

    // Mobile specific zoom control listeners
    const mobZoomIn = document.getElementById('mobile-btn-zoom-in');
    const mobZoomOut = document.getElementById('mobile-btn-zoom-out');
    const mobZoomReset = document.getElementById('mobile-btn-zoom-reset');

    if (mobZoomIn) {
        mobZoomIn.addEventListener('click', () => zoomCanvas(true));
    }
    if (mobZoomOut) {
        mobZoomOut.addEventListener('click', () => zoomCanvas(false));
    }
    if (mobZoomReset) {
        mobZoomReset.addEventListener('click', resetZoomAndPan);
    }
}

function setToolMode(mode) {
    state.toolMode = mode;
    const mobSelect = document.getElementById('mobile-tool-select');
    const mobDraw = document.getElementById('mobile-tool-draw');

    if (mode === 'select') {
        elements.toolSelect.classList.add('active');
        elements.toolDraw.classList.remove('active');
        if (mobSelect) mobSelect.classList.add('active');
        if (mobDraw) mobDraw.classList.remove('active');
        elements.modeIcon.className = "fa-solid fa-arrows-up-down-left-right";
        elements.modeText.textContent = "Select / Edit Mode";
    } else {
        elements.toolDraw.classList.add('active');
        elements.toolSelect.classList.remove('active');
        if (mobSelect) mobSelect.classList.remove('active');
        if (mobDraw) mobDraw.classList.add('active');
        elements.modeIcon.className = "fa-solid fa-square-plus";
        elements.modeText.textContent = "Draw Bounding Box";
        state.selectedBoxIndex = -1; // Deselect
        redraw();
    }
}

// Canvas interactions: zoom, pan, select, draw, drag, resize
function setupCanvasInteractions() {
    // 1. Mouse Wheel Zoom
    elements.canvasContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (!state.imgLoaded) return;

        const zoomIntensity = 0.12;
        const mouseX = e.clientX - elements.canvasContainer.getBoundingClientRect().left;
        const mouseY = e.clientY - elements.canvasContainer.getBoundingClientRect().top;

        // Calculate canvas coordinates of mouse pointer before zoom scale change
        const canvasMouseX = (mouseX - state.panX) / state.zoom;
        const canvasMouseY = (mouseY - state.panY) / state.zoom;

        // Update zoom scale factor
        if (e.deltaY < 0) {
            state.zoom = Math.min(state.zoom * (1 + zoomIntensity), 20.0); // max 20x zoom
        } else {
            state.zoom = Math.max(state.zoom * (1 - zoomIntensity), 0.05); // min 5% zoom
        }

        // Center zoom relative to mouse hover coordinates
        state.panX = mouseX - canvasMouseX * state.zoom;
        state.panY = mouseY - canvasMouseY * state.zoom;

        updateViewportTransform();
        redraw(); // redraw coordinates
    }, { passive: false });

    // 2. Mouse Down
    elements.canvasContainer.addEventListener('mousedown', (e) => {
        if (!state.imgLoaded) return;

        // Panning: Middle Mouse OR Left Mouse + Ctrl
        if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
            state.isPanning = true;
            state.panStartX = e.clientX - state.panX;
            state.panStartY = e.clientY - state.panY;
            elements.canvasContainer.style.cursor = 'grabbing';
            e.preventDefault();
            return;
        }

        if (e.button !== 0) return; // Only process left click

        const canvasPt = screenToCanvasCoords(e.clientX, e.clientY);

        if (state.toolMode === 'draw') {
            state.isDrawing = true;
            state.drawStart = { ...canvasPt };
            state.drawEnd = { ...canvasPt };
        }
        else if (state.toolMode === 'select') {
            const hit = getHitElement(canvasPt.x, canvasPt.y);

            if (hit) {
                if (hit.type === 'handle') {
                    // Start resize
                    state.dragMode = hit.mode;
                    state.dragStartBox = { ...state.boxes[state.selectedBoxIndex] };
                    state.dragStartMouse = { ...canvasPt };
                } else if (hit.type === 'box') {
                    // Select box, trigger drag move
                    state.selectedBoxIndex = hit.index;
                    selectClass(state.boxes[hit.index].class_id);
                    state.dragMode = 'move';
                    state.dragStartBox = { ...state.boxes[hit.index] };
                    state.dragStartMouse = { ...canvasPt };
                    redraw();
                }
            } else {
                // Deselect
                if (state.selectedBoxIndex !== -1) {
                    state.selectedBoxIndex = -1;
                    redraw();
                }
            }
        }
    });

    // 3. Mouse Move
    elements.canvasContainer.addEventListener('mousemove', (e) => {
        if (!state.imgLoaded) return;

        // Pan active
        if (state.isPanning) {
            state.panX = e.clientX - state.panStartX;
            state.panY = e.clientY - state.panStartY;
            updateViewportTransform();
            return;
        }

        const canvasPt = screenToCanvasCoords(e.clientX, e.clientY);

        // Draw active
        if (state.toolMode === 'draw' && state.isDrawing) {
            state.drawEnd = { ...canvasPt };
            redraw();
            return;
        }

        // Hover Cursor styling & Resize/Move handling
        if (state.toolMode === 'select') {
            if (state.dragMode) {
                // Handle Drag resizing/moving operations
                handleDragBox(canvasPt);
                state.hasUnsavedChanges = true;
                showStatus("Unsaved changes", "unsaved");
                redraw();
            } else {
                // Determine mouse pointer icons
                const hit = getHitElement(canvasPt.x, canvasPt.y);
                if (hit) {
                    if (hit.type === 'handle') {
                        if (hit.mode === 'nw' || hit.mode === 'se') elements.canvas.style.cursor = 'nwse-resize';
                        else if (hit.mode === 'ne' || hit.mode === 'sw') elements.canvas.style.cursor = 'nesw-resize';
                        else if (hit.mode === 'n' || hit.mode === 's') elements.canvas.style.cursor = 'ns-resize';
                        else if (hit.mode === 'e' || hit.mode === 'w') elements.canvas.style.cursor = 'ew-resize';
                    } else if (hit.type === 'box') {
                        elements.canvas.style.cursor = 'move';
                    }
                } else {
                    elements.canvas.style.cursor = 'default';
                }
            }
        }
    });

    // 4. Mouse Up
    window.addEventListener('mouseup', () => {
        if (state.isPanning) {
            state.isPanning = false;
            elements.canvasContainer.style.cursor = 'grab';
        }

        if (state.toolMode === 'draw' && state.isDrawing) {
            state.isDrawing = false;
            const created = createNewBox(state.drawStart.x, state.drawStart.y, state.drawEnd.x, state.drawEnd.y);
            if (created) {
                setToolMode('select');
            }
            redraw();
        }

        if (state.dragMode) {
            state.dragMode = null;
            state.dragStartBox = null;
            elements.canvas.style.cursor = 'default';
        }
    });

    // 5. Touch Start (Mobile Drawing and Interacting)
    elements.canvasContainer.addEventListener('touchstart', (e) => {
        if (!state.imgLoaded) return;

        if (e.touches.length === 2) {
            // Pinch-to-zoom & two-finger pan
            state.isDoubleTouching = true;
            const t1 = e.touches[0];
            const t2 = e.touches[1];
            state.touchStartDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
            state.touchStartMid = {
                x: (t1.clientX + t2.clientX) / 2,
                y: (t1.clientY + t2.clientY) / 2
            };
            state.panStartMidX = state.panX;
            state.panStartMidY = state.panY;
            state.zoomStartVal = state.zoom;
            e.preventDefault();
            return;
        }

        if (e.touches.length !== 1) return;

        const touch = e.touches[0];
        const canvasPt = screenToCanvasCoords(touch.clientX, touch.clientY);

        if (state.toolMode === 'draw') {
            state.isDrawing = true;
            state.drawStart = { ...canvasPt };
            state.drawEnd = { ...canvasPt };
            e.preventDefault();
        }
        else if (state.toolMode === 'select') {
            const hit = getHitElement(canvasPt.x, canvasPt.y);

            if (hit) {
                if (hit.type === 'handle') {
                    // Start resize
                    state.dragMode = hit.mode;
                    state.dragStartBox = { ...state.boxes[state.selectedBoxIndex] };
                    state.dragStartMouse = { ...canvasPt };
                } else if (hit.type === 'box') {
                    // Select box, trigger drag move
                    state.selectedBoxIndex = hit.index;
                    selectClass(state.boxes[hit.index].class_id);
                    state.dragMode = 'move';
                    state.dragStartBox = { ...state.boxes[hit.index] };
                    state.dragStartMouse = { ...canvasPt };
                    redraw();
                }
                e.preventDefault();
            } else {
                // Drag empty space to pan on mobile
                state.isPanning = true;
                state.panStartX = touch.clientX - state.panX;
                state.panStartY = touch.clientY - state.panY;

                if (state.selectedBoxIndex !== -1) {
                    state.selectedBoxIndex = -1;
                    redraw();
                }
                e.preventDefault();
            }
        }
    }, { passive: false });

    // 6. Touch Move
    elements.canvasContainer.addEventListener('touchmove', (e) => {
        if (!state.imgLoaded) return;

        if (state.isDoubleTouching && e.touches.length === 2) {
            const t1 = e.touches[0];
            const t2 = e.touches[1];
            const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
            const mid = {
                x: (t1.clientX + t2.clientX) / 2,
                y: (t1.clientY + t2.clientY) / 2
            };

            // Calculate zoom scale
            const scale = dist / state.touchStartDist;
            state.zoom = Math.max(0.05, Math.min(state.zoomStartVal * scale, 20.0));

            // Calculate pan dx & dy
            const dx = mid.x - state.touchStartMid.x;
            const dy = mid.y - state.touchStartMid.y;

            state.panX = state.panStartMidX + dx;
            state.panY = state.panStartMidY + dy;

            updateViewportTransform();
            redraw();
            e.preventDefault();
            return;
        }

        if (e.touches.length !== 1) return;

        const touch = e.touches[0];
        const canvasPt = screenToCanvasCoords(touch.clientX, touch.clientY);

        if (state.isPanning) {
            state.panX = touch.clientX - state.panStartX;
            state.panY = touch.clientY - state.panStartY;
            updateViewportTransform();
            e.preventDefault();
            return;
        }

        if (state.toolMode === 'draw' && state.isDrawing) {
            state.drawEnd = { ...canvasPt };
            redraw();
            e.preventDefault();
            return;
        }

        if (state.toolMode === 'select' && state.dragMode) {
            handleDragBox(canvasPt);
            state.hasUnsavedChanges = true;
            showStatus("Unsaved changes", "unsaved");
            redraw();
            e.preventDefault();
        }
    }, { passive: false });

    // 7. Touch End
    elements.canvasContainer.addEventListener('touchend', (e) => {
        if (state.isDoubleTouching) {
            if (e.touches.length < 2) {
                state.isDoubleTouching = false;
            }
            e.preventDefault();
            return;
        }

        if (state.isPanning) {
            state.isPanning = false;
            e.preventDefault();
        }

        if (state.toolMode === 'draw' && state.isDrawing) {
            state.isDrawing = false;
            const created = createNewBox(state.drawStart.x, state.drawStart.y, state.drawEnd.x, state.drawEnd.y);
            if (created) {
                setToolMode('select');
            }
            redraw();
            e.preventDefault();
        }

        if (state.dragMode) {
            state.dragMode = null;
            state.dragStartBox = null;
            e.preventDefault();
        }
    }, { passive: false });
}

// Bounding box dragging/resizing calculation helper
function handleDragBox(canvasPt) {
    const box = state.boxes[state.selectedBoxIndex];
    const original = state.dragStartBox;
    const imgW = state.img.naturalWidth;
    const imgH = state.img.naturalHeight;

    // Convert original normalized box dimensions to physical canvas pixels
    const origW = original.width * imgW;
    const origH = original.height * imgH;
    let origX1 = original.x_center * imgW - origW / 2;
    let origY1 = original.y_center * imgH - origH / 2;
    let origX2 = origX1 + origW;
    let origY2 = origY1 + origH;

    // Mouse deltas
    const dx = canvasPt.x - state.dragStartMouse.x;
    const dy = canvasPt.y - state.dragStartMouse.y;

    if (state.dragMode === 'move') {
        const xc = (original.x_center * imgW) + dx;
        const yc = (original.y_center * imgH) + dy;
        // Limit movement within bounds
        const hW = origW / 2;
        const hH = origH / 2;
        const xcBounded = Math.max(hW, Math.min(xc, imgW - hW));
        const ycBounded = Math.max(hH, Math.min(yc, imgH - hH));

        box.x_center = xcBounded / imgW;
        box.y_center = ycBounded / imgH;
    }
    else {
        // Resizing logic
        let x1 = origX1;
        let y1 = origY1;
        let x2 = origX2;
        let y2 = origY2;

        if (state.dragMode.includes('w')) x1 += dx;
        if (state.dragMode.includes('e')) x2 += dx;
        if (state.dragMode.includes('n')) y1 += dy;
        if (state.dragMode.includes('s')) y2 += dy;

        // Ensure box dimensions don't invert (min size constraint)
        const minDim = 4;
        if (x2 - x1 < minDim) {
            if (state.dragMode.includes('w')) x1 = x2 - minDim;
            if (state.dragMode.includes('e')) x2 = x1 + minDim;
        }
        if (y2 - y1 < minDim) {
            if (state.dragMode.includes('n')) y1 = y2 - minDim;
            if (state.dragMode.includes('s')) y2 = y1 + minDim;
        }

        saveYoloBoxCoords(state.selectedBoxIndex, x1, y1, x2, y2);
    }
}

// Keydown Events (Hotkeys)
function handleKeyDown(e) {
    // If user typing in input/search elements, ignore shortcuts
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') {
        return;
    }

    const key = e.key.toLowerCase();

    // Ctrl + S (Save)
    if ((e.ctrlKey || e.metaKey) && key === 's') {
        e.preventDefault();
        saveAnnotations();
        return;
    }

    // Navigation: A (Prev), D (Next)
    if (key === 'a' || e.key === 'ArrowLeft') {
        e.preventDefault();
        if (state.currentImageIndex > 0) {
            if (state.hasUnsavedChanges) {
                saveAnnotations().then(() => loadImageIndex(state.currentImageIndex - 1));
            } else {
                loadImageIndex(state.currentImageIndex - 1);
            }
        }
    }
    else if (key === 'd' || e.key === 'ArrowRight') {
        e.preventDefault();
        if (state.currentImageIndex < state.images.length - 1) {
            if (state.hasUnsavedChanges) {
                saveAnnotations().then(() => loadImageIndex(state.currentImageIndex + 1));
            } else {
                loadImageIndex(state.currentImageIndex + 1);
            }
        }
    }

    // Tools switching
    else if (key === 'r') {
        setToolMode('draw');
    }
    else if (key === 'v') {
        setToolMode('select');
    }

    // Delete Box
    else if (e.key === 'e' || e.key === 'Delete') {
        if (state.selectedBoxIndex !== -1 && state.toolMode === 'select') {
            state.boxes.splice(state.selectedBoxIndex, 1);
            state.selectedBoxIndex = -1;
            state.hasUnsavedChanges = true;
            showStatus("Unsaved changes", "unsaved");
            redraw();
        }
    }

    // Escape (Cancel drawing, deselect)
    else if (e.key === 'Escape') {
        if (state.isDrawing) {
            state.isDrawing = false;
        }
        if (state.selectedBoxIndex !== -1) {
            state.selectedBoxIndex = -1;
        }
        setToolMode('select');
        redraw();
    }

    // Class selection hotkeys: checks custom shortcuts first, then defaults to digit match
    else {
        let matchedClassId = null;

        // 1. Check custom class_shortcuts
        if (state.classShortcuts) {
            Object.entries(state.classShortcuts).forEach(([classId, shortcutKey]) => {
                if (key === shortcutKey.toLowerCase()) {
                    matchedClassId = parseInt(classId);
                }
            });
        }

        // 2. Fallback to default index digits
        if (matchedClassId === null && /^\d$/.test(key)) {
            const num = parseInt(key);
            if (state.classes.hasOwnProperty(num)) {
                matchedClassId = num;
            }
        }

        // Apply class selection change
        if (matchedClassId !== null) {
            selectClass(matchedClassId);
            // If box is selected, update its class ID directly
            if (state.selectedBoxIndex !== -1 && state.toolMode === 'select') {
                state.boxes[state.selectedBoxIndex].class_id = matchedClassId;
                state.hasUnsavedChanges = true;
                showStatus("Unsaved changes", "unsaved");
                redraw();
            }
        }
    }
}

// Show saving status message
function showStatus(msg, type) {
    elements.saveStatus.textContent = msg;
    elements.saveStatus.className = ''; // Reset

    if (type === 'idle') {
        elements.saveStatus.classList.add('save-status-idle');
        elements.saveStatus.innerHTML = `<i class="fa-solid fa-check-double"></i> ${msg}`;
    } else if (type === 'saving') {
        elements.saveStatus.classList.add('save-status-saving');
        elements.saveStatus.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${msg}`;
    } else if (type === 'unsaved') {
        elements.saveStatus.classList.add('save-status-unsaved');
        elements.saveStatus.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> ${msg}`;
    }
}

// Upload files/folders of local images to active directory
async function handleMultipleFilesUpload(files) {
    if (!files || files.length === 0) return;

    // Auto save if there are unsaved changes
    if (state.hasUnsavedChanges) {
        showStatus("Saving current annotations before upload...", "saving");
        await saveAnnotations();
    }

    let successCount = 0;
    let failCount = 0;

    showStatus(`Uploading ${files.length} items...`, "saving");

    const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];

        if (file.name.startsWith('.')) continue; // skip hidden

        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!validExtensions.includes(ext)) {
            continue; // skip non-images
        }

        const formData = new FormData();
        formData.append('file', file);

        const relPath = file.webkitRelativePath || '';
        if (relPath) {
            formData.append('relativePath', relPath);
        }

        try {
            const response = await fetch(`/api/upload/${state.currentSubpath}`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                successCount++;
            } else {
                failCount++;
            }
        } catch (err) {
            console.error("Upload error", err);
            failCount++;
        }

        showStatus(`Uploading: ${successCount + failCount}/${files.length} items...`, "saving");
    }

    showStatus(`Upload complete! Succeeded: ${successCount}, Failed: ${failCount}`, "idle");

    // Reload tree structure and list
    await fetchTree();
    await fetchImages(true);
}

// Start application
window.onload = init;
