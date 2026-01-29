/**
 * Ring Test Module Controller
 */

let currentDiameter = 12;
let currentFile = null;

document.addEventListener('DOMContentLoaded', () => {
    const ringElements = {
        uploadZone: document.getElementById('ringUploadZone'),
        fileInput: document.getElementById('ringFileInput'),
        preview: document.getElementById('ringPreview'),
        previewImage: document.getElementById('ringPreviewImage'),
        analyzeBtn: document.getElementById('ringAnalyzeBtn'),
        resultsContent: document.getElementById('ringResultsContent'),
        resultsVisual: document.getElementById('ringResultsVisual'),
        placeholderData: document.getElementById('ringPlaceholderData'),
        placeholderVisual: document.getElementById('ringPlaceholderVisual'),
        statusText: document.getElementById('ringStatusText'),
        statusReason: document.getElementById('ringStatusReason'),
        statusIcon: document.getElementById('ringStatusIcon'),
        level1Results: document.getElementById('ringLevel1Results'),
        level2Results: document.getElementById('ringLevel2Results'),
        debugImage: document.getElementById('ringDebugImage')
    };

    // 1. Diameter Selection
    document.querySelectorAll('#ringDiameterOptions .diameter-card').forEach(card => {
        card.onclick = () => {
            document.querySelectorAll('#ringDiameterOptions .diameter-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            currentDiameter = parseFloat(card.dataset.diameter);
        };
    });

    // 2. File Upload
    Common.initFileUpload(ringElements, (file) => {
        Common.openCropper(file, 'ringPreviewImage', (croppedFile) => {
            currentFile = croppedFile;
            ringElements.uploadZone.classList.add('hidden');
            ringElements.preview.classList.remove('hidden');
            ringElements.analyzeBtn.disabled = false;
        });
    });

    // 3. Remove Image
    document.getElementById('ringRemoveImage').onclick = () => {
        currentFile = null;
        ringElements.uploadZone.classList.remove('hidden');
        ringElements.preview.classList.add('hidden');
        ringElements.analyzeBtn.disabled = true;
        ringElements.resultsContent.classList.add('hidden');
        ringElements.resultsVisual.classList.add('hidden');
        ringElements.placeholderData.classList.remove('hidden');
        ringElements.placeholderVisual.classList.remove('hidden');
    };

    // 4. Analyze
    ringElements.analyzeBtn.onclick = () => {
        if (!currentFile) return;

        const upscale = document.getElementById('upscaleCheckbox')?.checked || false;
        const edge_segment = document.getElementById('edgeSegmentCheckbox')?.checked || false;

        console.log('AI Upscale:', upscale);
        console.log('Edge Segmentation:', edge_segment);

        Common.runAnalysis('/api/ring-test', currentFile, currentDiameter, (data) => {
            displayResults(data, ringElements);
        }, { upscale, edge_segment });
    };
});

function displayResults(data, elements) {
    const isPass = data.status === 'PASS';
    Common.updateStatusUI(elements, isPass, data.reason);

    // Level 1: Qualitative
    if (data.level1 && elements.level1Results) {
        elements.level1Results.innerHTML = `
            <div class="space-y-1">
                ${Common.createMetricRow('Layers Detected', data.level1.regions_visible)}
                ${Common.createMetricRow('Outer Ring', data.level1.ring_continuous)}
                ${Common.createMetricRow('Concentricity', data.level1.concentric)}
                ${Common.createMetricRow('Thickness Uniformity', data.level1.thickness_uniform)}
            </div>
        `;
    }

    // Level 2: Dimensional
    if (data.level2 && elements.level2Results) {
        const t = data.level2.thickness_mm;
        const low = currentDiameter * 0.07;
        const high = currentDiameter * 0.10;
        const inRange = t >= low && t <= high;

        elements.level2Results.innerHTML = `
            <div class="space-y-4">
                <div class="bg-white/5 rounded-xl p-4 space-y-2 border border-white/5">
                    <p class="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">Observations (in mm)</p>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-gray-400">Diameter of rebar, D</span>
                        <span class="text-sm text-white font-bold">${currentDiameter} mm</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-gray-400">Avg Thickness, t<sub>TM</sub></span>
                        <span class="text-sm text-white font-bold">${t.toFixed(2)} mm</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-gray-400">Thickness Range</span>
                        <span class="text-sm text-blue-400 font-bold">${data.level2.thickness_mm_min.toFixed(2)} — ${data.level2.thickness_mm_max.toFixed(2)} mm</span>
                    </div>
                    <div class="mt-4 pt-4 border-t border-white/5">
                        <div class="flex justify-between items-center bg-blue-500/10 p-4 rounded-xl border border-blue-500/20">
                            <div class="flex flex-col">
                                <span class="text-[9px] text-blue-300 font-black uppercase tracking-widest mb-0.5">Target Window</span>
                                <span class="text-[10px] text-gray-500 font-medium italic">Requirement: 0.07D — 0.10D</span>
                            </div>
                            <span class="text-lg font-black text-blue-400 tracking-tighter">${low.toFixed(2)} — ${high.toFixed(2)} <span class="text-[10px] text-blue-500/50 ml-0.5">mm</span></span>
                        </div>
                    </div>
                </div>

                <div class="space-y-1">
                    <p class="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">L2 Acceptance Criteria</p>
                    ${Common.createMetricRow(`Is ${t.toFixed(2)}mm ≥ ${low.toFixed(2)}mm ?`, t >= low)}
                    ${Common.createMetricRow(`Is ${t.toFixed(2)}mm ≤ ${high.toFixed(2)}mm ?`, t <= high)}
                </div>

                <div class="pt-4 border-t border-white/5 flex justify-between items-center">
                    <span class="text-[9px] text-gray-500 font-bold uppercase tracking-widest">Decision</span>
                    <span class="px-3 py-1 rounded bg-${isPass ? 'green' : 'red'}-500/10 text-${isPass ? 'green' : 'red'}-400 text-[10px] font-black uppercase tracking-widest border border-${isPass ? 'green' : 'red'}-500/20">
                        ${isPass ? 'ACCEPT REBAR LOT' : 'REJECT REBAR LOT'}
                    </span>
                </div>
            </div>
        `;
    }

    // Debug Image
    if (data.debug_image_url && elements.debugImage) {
        elements.debugImage.src = data.debug_image_url + '?t=' + Date.now();
    }

    // Segmented Image (if edge segmentation was used)
    const segmentedContainer = document.getElementById('ringSegmentedImageContainer');
    const segmentedImage = document.getElementById('ringSegmentedImage');

    if (data.segmented_image_url && segmentedImage) {
        segmentedImage.src = data.segmented_image_url;
        segmentedContainer.classList.remove('hidden');
        console.log('✅ Displaying edge-segmented image');
    } else {
        segmentedContainer.classList.add('hidden');
    }
}
