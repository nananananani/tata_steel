// Tata Steel TMT Analysis - JavaScript
console.log('🚀 App script loaded!');

// Global State
let currentSection = 'landingPage';
let ringDiameter = 12;
let ribDiameter = 12;
let ringFile = null;
let ribFile = null;
let cropper = null;
let cropperTarget = null; // 'ring' or 'rib'

// Global Elements (initialized in DOMContentLoaded)
let ringElements = {};
let ribElements = {};
let loadingOverlay = null;

// Navigation - Defined globally immediately
window.showSection = function (sectionId) {
    console.log('Navigating to:', sectionId);

    // Validate section exists
    const section = document.getElementById(sectionId);
    if (!section) {
        console.error('Section not found:', sectionId);
        return;
    }

    // Hide all sections
    ['landingPage', 'ringTestPage', 'ribTestPage'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add('hidden');
    });

    // Show target
    section.classList.remove('hidden');
    currentSection = sectionId;
    window.scrollTo(0, 0);
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ DOM Content Loaded');

    try {
        // Initialize Elements
        loadingOverlay = document.getElementById('loadingOverlay');

        initializeRingElements();
        initializeRibElements();

        setupNavigation(); // Add click listeners for non-onclick elements if any
        setupRingTest();
        setupRibTest();

        console.log('✨ App initialized successfully');

    } catch (error) {
        console.error('❌ Initialization failed:', error);
        alert('App initialization failed. See console for details: ' + error.message);
    }
});

function initializeRingElements() {
    const ids = [
        'ringUploadZone', 'ringFileInput', 'ringPreview', 'ringPreviewImage',
        'ringAnalyzeBtn', 'ringResultsContent', 'ringResultsVisual',
        'ringPlaceholderData', 'ringPlaceholderVisual', 'ringDebugImage',
        'ringStatusIcon', 'ringStatusText', 'ringStatusReason',
        'ringLevel1Results', 'ringLevel2Results'
    ];

    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) console.warn(`Element not found: ${id}`);
        // Map 'ringUploadZone' -> 'uploadZone' for cleaner internal usage, or keep map
        // Let's just key by the ID directly or a simple name
        // For simplicity with previous code, let's map manually
    });

    ringElements = {
        uploadZone: document.getElementById('ringUploadZone'),
        fileInput: document.getElementById('ringFileInput'),
        preview: document.getElementById('ringPreview'),
        previewImage: document.getElementById('ringPreviewImage'),
        removeImageBtn: document.getElementById('ringRemoveImage'),
        analyzeBtn: document.getElementById('ringAnalyzeBtn'),
        resultsContent: document.getElementById('ringResultsContent'),
        resultsVisual: document.getElementById('ringResultsVisual'),
        placeholderData: document.getElementById('ringPlaceholderData'),
        placeholderVisual: document.getElementById('ringPlaceholderVisual'),
        debugImage: document.getElementById('ringDebugImage'),
        statusIcon: document.getElementById('ringStatusIcon'),
        statusText: document.getElementById('ringStatusText'),
        statusReason: document.getElementById('ringStatusReason'),
        level1Results: document.getElementById('ringLevel1Results'),
        level2Results: document.getElementById('ringLevel2Results')
    };
}

function initializeRibElements() {
    ribElements = {
        uploadZone: document.getElementById('ribUploadZone'),
        fileInput: document.getElementById('ribFileInput'),
        preview: document.getElementById('ribPreview'),
        previewImage: document.getElementById('ribPreviewImage'),
        analyzeBtn: document.getElementById('ribAnalyzeBtn'),
        resultsContent: document.getElementById('ribResultsContent'),
        resultsVisual: document.getElementById('ribResultsVisual'),
        placeholderData: document.getElementById('ribPlaceholderData'),
        placeholderVisual: document.getElementById('ribPlaceholderVisual'),
        debugImage: document.getElementById('ribDebugImage'),
        statusIcon: document.getElementById('ribStatusIcon'),
        statusText: document.getElementById('ribStatusText'),
        statusReason: document.getElementById('ribStatusReason'),
        arValue: document.getElementById('arValueDisplay'),
        parameters: document.getElementById('ribParameters')
    };
}

// ==========================================
// RING TEST LOGIC
// ==========================================
function setupRingTest() {
    if (!ringElements.uploadZone) return; // Safety check

    // Diameter Selection
    const options = document.querySelectorAll('#ringDiameterOptions .diameter-card');
    if (options) {
        options.forEach(card => {
            card.addEventListener('click', () => {
                options.forEach(c => c.classList.remove('selected', 'border-blue-500', 'bg-blue-500/10'));
                card.classList.add('selected', 'border-blue-500', 'bg-blue-500/10');
                ringDiameter = parseInt(card.dataset.diameter);
            });
        });
    }

    // File Upload
    setupFileUpload(ringElements, file => {
        ringFile = file;
        if (ringElements.analyzeBtn) ringElements.analyzeBtn.disabled = false;
    });

    // Remove Image
    if (ringElements.removeImageBtn) {
        ringElements.removeImageBtn.addEventListener('click', () => {
            ringFile = null;
            ringElements.fileInput.value = '';
            ringElements.preview.classList.add('hidden');
            ringElements.uploadZone.classList.remove('hidden');
            ringElements.analyzeBtn.disabled = true;

            // Hide results
            ringElements.resultsContent.classList.add('hidden');
            ringElements.resultsVisual.classList.add('hidden');
            ringElements.placeholderData.classList.remove('hidden');
            ringElements.placeholderVisual.classList.remove('hidden');
        });
    }

    // Analyze Action
    if (ringElements.analyzeBtn) {
        ringElements.analyzeBtn.addEventListener('click', async () => {
            if (!ringFile) return;
            await runAnalysis('/api/ring-test', ringFile, ringDiameter, displayRingResults);
        });
    }
}

function displayRingResults(result) {
    // Status
    const isPass = result.status === 'PASS';
    updateStatus(ringElements, isPass, result.reason);

    // Level 1: Color & Shape Analysis
    if (result.level1 && ringElements.level1Results) {
        ringElements.level1Results.innerHTML = `
            <div class="space-y-4">
                <div class="bg-gray-800/30 rounded-xl p-4 space-y-3 border border-gray-700/50">
                    ${createMetricRow('Layers Detected', result.level1.regions_visible)}
                    ${createMetricRow('Outer Ring', result.level1.ring_continuous)}
                    ${createMetricRow('Concentricity', result.level1.concentric)}
                    ${createMetricRow('Thickness Uniformity', result.level1.thickness_uniform)}
                </div>
            </div>
        `;
    }

    // Level 2: Dimensional Analysis
    if (result.level2 && ringElements.level2Results) {
        const t = result.level2.thickness_mm;
        const low = ringDiameter * 0.07;
        const high = ringDiameter * 0.10;

        ringElements.level2Results.innerHTML = `
            <div class="space-y-6">
                <!-- Observations -->
                <div>
                     <p class="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">Observations (in mm)</p>
                     <div class="space-y-2 bg-gray-800/30 rounded-xl p-4 border border-gray-700/50">
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-gray-400">Diameter of rebar, D</span>
                            <span class="text-white font-bold">${ringDiameter} mm</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-gray-400">Measured thickness, t<sub>TM</sub></span>
                            <span class="text-white font-bold">${t.toFixed(2)} mm</span>
                        </div>
                     </div>
                </div>

                <!-- Acceptance Criteria -->
                <div>
                     <p class="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">L2 Acceptance Criteria</p>
                     <div class="space-y-3 bg-gray-800/30 rounded-xl p-4 border border-gray-700/50">
                        ${createMetricRow(`Is ${t.toFixed(2)}mm ≥ ${low.toFixed(2)}mm ?`, t >= low)}
                        ${createMetricRow(`Is ${t.toFixed(2)}mm ≤ ${high.toFixed(2)}mm ?`, t <= high)}
                     </div>
                </div>

                <!-- Final Decision -->
                <div class="pt-4 border-t border-gray-800 flex justify-between items-center">
                    <span class="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Decision</span>
                    <span class="px-3 py-1 rounded bg-${isPass ? 'green' : 'red'}-500/10 text-${isPass ? 'green' : 'red'}-400 text-[10px] font-black uppercase tracking-widest border border-${isPass ? 'green' : 'red'}-500/20">
                        ${isPass ? 'Accept Rebar Lot' : 'Reject Rebar Lot'}
                    </span>
                </div>
            </div>
        `;
    }

    // Debug Image
    if (result.debug_image_url && ringElements.debugImage) {
        ringElements.debugImage.src = result.debug_image_url + '?t=' + Date.now();
    }
}

// ==========================================
// RIB TEST LOGIC
// ==========================================
function setupRibTest() {
    if (!ribElements.uploadZone) return;

    // Diameter Selection
    const options = document.querySelectorAll('#ribDiameterOptions .diameter-card');
    if (options) {
        options.forEach(card => {
            card.addEventListener('click', () => {
                options.forEach(c => c.classList.remove('selected', 'border-purple-500', 'bg-purple-500/10'));
                card.classList.add('selected', 'border-purple-500', 'bg-purple-500/10');
                ribDiameter = parseInt(card.dataset.diameter);
            });
        });
    }

    // File Upload
    setupFileUpload(ribElements, file => {
        openCropper(file, 'rib');
    });

    // Analyze Action
    if (ribElements.analyzeBtn) {
        ribElements.analyzeBtn.addEventListener('click', async () => {
            if (!ribFile) return;
            await runAnalysis('/api/rib-test', ribFile, ribDiameter, displayRibResults);
        });
    }
}

// ==========================================
// CROPPER SYSTEM
// ==========================================
function openCropper(file, target) {
    const modal = document.getElementById('cropperModal');
    const image = document.getElementById('cropperImage');

    if (!modal || !image) return;

    cropperTarget = target;
    const reader = new FileReader();
    reader.onload = (e) => {
        image.src = e.target.result;
        modal.classList.remove('hidden');

        if (cropper) cropper.destroy();

        cropper = new Cropper(image, {
            viewMode: 2,
            dragMode: 'move',
            autoCropArea: 0.8,
            restore: false,
            guides: true,
            center: true,
            highlight: false,
            cropBoxMovable: true,
            cropBoxResizable: true,
            toggleDragModeOnDblclick: false,
        });
    };
    reader.readAsDataURL(file);
}

// Cropper UI Controls
document.getElementById('saveCrop')?.addEventListener('click', () => {
    if (!cropper) return;

    cropper.getCroppedCanvas({
        maxWidth: 2000,
        maxHeight: 2000,
        fillColor: '#000',
    }).toBlob((blob) => {
        const file = new File([blob], "cropped_rebar.jpg", { type: "image/jpeg" });

        if (cropperTarget === 'rib') {
            ribFile = file;
            ribElements.previewImage.src = URL.createObjectURL(blob);
            ribElements.uploadZone.classList.add('hidden');
            ribElements.preview.classList.remove('hidden');
            ribElements.analyzeBtn.disabled = false;
        } else {
            ringFile = file;
            ringElements.previewImage.src = URL.createObjectURL(blob);
            ringElements.uploadZone.classList.add('hidden');
            ringElements.preview.classList.remove('hidden');
            ringElements.analyzeBtn.disabled = false;
        }

        closeCropper();
    }, 'image/jpeg', 0.9);
});

document.getElementById('cancelCrop')?.addEventListener('click', closeCropper);
document.getElementById('rotateLeft')?.addEventListener('click', () => cropper?.rotate(-90));
document.getElementById('zoomIn')?.addEventListener('click', () => cropper?.zoom(0.1));
document.getElementById('resetCrop')?.addEventListener('click', () => cropper?.reset());

function closeCropper() {
    document.getElementById('cropperModal').classList.add('hidden');
    if (cropper) {
        cropper.destroy();
        cropper = null;
    }
}

function displayRibResults(result) {
    const isPass = result.status === 'PASS';
    updateStatus(ribElements, isPass, result.reason);

    // AR Value
    if (result.ar_value !== undefined && ribElements.arValue) {
        ribElements.arValue.textContent = result.ar_value.toFixed(4);
    }

    // Parameters
    if (result.rib_count !== undefined && ribElements.parameters) {
        ribElements.parameters.innerHTML = `
            <div class="bg-gray-800/50 rounded-xl p-4 space-y-3">
                <div class="flex justify-between">
                     <span class="text-gray-400 text-sm">No. of Ribs</span>
                     <span class="text-white font-bold">${result.rib_count}</span>
                </div>
                <div class="flex justify-between">
                     <span class="text-gray-400 text-sm">Avg Length</span>
                     <span class="text-white font-bold">${result.avg_length_mm} mm</span>
                </div>
                <div class="flex justify-between">
                     <span class="text-gray-400 text-sm">Avg Angle</span>
                     <span class="text-white font-bold">${result.avg_angle_deg}°</span>
                </div>
                <div class="flex justify-between">
                     <span class="text-gray-400 text-sm">Avg Height</span>
                     <span class="text-white font-bold">${result.avg_height_mm} mm</span>
                </div>
                <div class="flex justify-between">
                     <span class="text-gray-400 text-sm">Avg Spacing</span>
                     <span class="text-white font-bold">${result.avg_spacing_mm} mm</span>
                </div>
            </div>
        `;
    }

    // Debug Image
    if (result.debug_image_url && ribElements.debugImage) {
        ribElements.debugImage.src = result.debug_image_url + '?t=' + Date.now();
    }
}

// ==========================================
// SHARED UTILS
// ==========================================
function setupNavigation() {
    // Add any specific listeners if needed
}

function setupFileUpload(elements, onFileSelected) {
    const { uploadZone, fileInput, preview, previewImage } = elements;

    if (!uploadZone || !fileInput) {
        console.error('Missing upload elements', elements);
        return;
    }

    uploadZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', e => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    uploadZone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadZone.classList.add('border-white');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('border-white');
    });

    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.classList.remove('border-white');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = e => {
            if (previewImage) previewImage.src = e.target.result;
            if (uploadZone) uploadZone.classList.add('hidden');
            if (preview) preview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

        onFileSelected(file);
    }
}

async function runAnalysis(endpoint, file, diameter, callback) {
    if (loadingOverlay) loadingOverlay.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('diameter', diameter);

        const res = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(await res.text());

        const data = await res.json();

        // Show results containers
        const els = endpoint.includes('ring') ? ringElements : ribElements;
        if (els.resultsContent) els.resultsContent.classList.remove('hidden');
        if (els.resultsVisual) els.resultsVisual.classList.remove('hidden');
        if (els.placeholderData) els.placeholderData.classList.add('hidden');
        if (els.placeholderVisual) els.placeholderVisual.classList.add('hidden');

        callback(data);

    } catch (e) {
        console.error(e);
        alert('Analysis Failed: ' + e.message);
    } finally {
        if (loadingOverlay) loadingOverlay.classList.add('hidden');
    }
}

function updateStatus(elements, isPass, reason) {
    const color = isPass ? 'text-green-400' : 'text-red-400';
    const bg = isPass ? 'bg-green-500/20' : 'bg-red-500/20';

    if (elements.statusText) {
        elements.statusText.textContent = isPass ? 'PASS' : 'FAIL';
        elements.statusText.className = `text-2xl font-bold ${color}`;
    }

    if (elements.statusReason) elements.statusReason.textContent = reason;

    if (elements.statusIcon) {
        elements.statusIcon.className = `w-12 h-12 rounded-full flex items-center justify-center ${bg} mr-6`;
        elements.statusIcon.innerHTML = isPass
            ? `<svg class="w-8 h-8 ${color}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`
            : `<svg class="w-8 h-8 ${color}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>`;
    }
}

function createMetricRow(label, passed) {
    const color = passed ? 'text-green-400' : 'text-red-400';
    const icon = passed
        ? '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>'
        : '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';

    return `
        <div class="flex justify-between items-center">
            <span class="text-gray-300 text-sm">${label}</span>
            <span class="flex items-center gap-2 ${color} font-bold">
                ${passed ? 'YES' : 'NO'}
                ${icon}
            </span>
        </div>
    `;
}
