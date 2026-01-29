/**
 * Common Utilities for Tata Steel Rebar Analysis Suite
 */

const Common = {
    cropper: null,

    /**
     * Initialize File Upload logic
     */
    initFileUpload: (elements, onFileSelected) => {
        const { uploadZone, fileInput, preview, previewImage } = elements;

        uploadZone.onclick = () => fileInput.click();

        fileInput.onchange = (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        };

        uploadZone.ondragover = (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        };

        uploadZone.ondragleave = () => {
            uploadZone.classList.remove('dragover');
        };

        uploadZone.ondrop = (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFile(e.dataTransfer.files[0]);
            }
        };

        function handleFile(file) {
            if (!file || !file.type.startsWith('image/')) {
                alert('Please upload a valid image file');
                return;
            }
            onFileSelected(file);
        }
    },

    /**
     * Cropper System
     */
    openCropper: (file, targetImageId, onSave) => {
        const modal = document.getElementById('cropperModal');
        const cropperImg = document.getElementById('cropperImage');
        const reader = new FileReader();

        reader.onload = (e) => {
            cropperImg.src = e.target.result;
            modal.style.display = 'flex';

            if (Common.cropper) Common.cropper.destroy();

            Common.cropper = new Cropper(cropperImg, {
                viewMode: 2,
                autoCropArea: 0.9,
                responsive: true,
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

        document.getElementById('saveCrop').onclick = () => {
            const canvas = Common.cropper.getCroppedCanvas({ maxWidth: 2048, maxHeight: 2048 });
            canvas.toBlob((blob) => {
                const croppedFile = new File([blob], "sample.jpg", { type: "image/jpeg" });
                const url = URL.createObjectURL(blob);

                const previewImg = document.getElementById(targetImageId);
                if (previewImg) previewImg.src = url;

                onSave(croppedFile, url);
                Common.closeCropper();
            }, 'image/jpeg', 0.95);
        };

        document.getElementById('cancelCrop').onclick = Common.closeCropper;
    },

    closeCropper: () => {
        document.getElementById('cropperModal').style.display = 'none';
        if (Common.cropper) {
            Common.cropper.destroy();
            Common.cropper = null;
        }
    },

    /**
     * API Call Wrapper
     */
    runAnalysis: async (endpoint, file, diameter, onResult, extraParams = {}) => {
        const loader = document.getElementById('loadingOverlay');
        if (loader) loader.style.display = 'flex';

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('diameter', diameter);

            // Append extra parameters (e.g., upscale)
            for (const key in extraParams) {
                formData.append(key, extraParams[key]);
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(await response.text());

            const data = await response.json();
            onResult(data);
        } catch (error) {
            console.error('Analysis Error:', error);
            alert('Analysis failed: ' + error.message);
        } finally {
            if (loader) loader.style.display = 'none';
        }
    },

    createMetricRow: (label, passed, customValue = null) => {
        const colorClass = passed ? 'text-green-400' : 'text-red-400';
        const icon = passed
            ? '<svg class="w-4 h-4 checkmark" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>'
            : '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M6 18L18 6M6 6l12 12"></path></svg>';

        return `
            <div class="metric-row">
                <div class="flex items-center">
                    <div class="w-1.5 h-1.5 rounded-full bg-blue-500/50 mr-3"></div>
                    <span class="text-gray-300 text-sm">${label}</span>
                </div>
                <div class="flex items-center gap-3">
                    <span class="text-[10px] font-black uppercase tracking-widest ${colorClass}">${customValue || (passed ? 'YES' : 'NO')}</span>
                    <div class="w-6 h-6 rounded-full flex items-center justify-center bg-white/5 border border-white/10 ${colorClass}">
                        ${icon}
                    </div>
                </div>
            </div>
        `;
    },

    updateStatusUI: (elements, isPass, reason) => {
        const { statusText, statusReason, statusIcon, resultsContent, resultsVisual, placeholderData, placeholderVisual } = elements;

        if (resultsContent) resultsContent.classList.remove('hidden');
        if (resultsVisual) resultsVisual.classList.remove('hidden');
        if (placeholderData) placeholderData.classList.add('hidden');
        if (placeholderVisual) placeholderVisual.classList.add('hidden');

        if (statusText) {
            statusText.textContent = isPass ? 'PASS' : 'FAIL';
            statusText.className = `text-2xl font-black ${isPass ? 'text-green-400' : 'text-red-400'}`;
        }

        if (statusReason) statusReason.textContent = reason || (isPass ? 'ALL CRITERIA MET' : 'CRITERIA NOT MET');

        if (statusIcon) {
            const color = isPass ? 'text-green-400' : 'text-red-400';
            const shadow = isPass ? 'shadow-green-500/20' : 'shadow-red-500/20';
            const bg = isPass ? 'bg-green-500/10' : 'bg-red-500/10';
            const border = isPass ? 'border-green-500/20' : 'border-red-500/20';

            statusIcon.innerHTML = `
                <div class="${bg} ${border} ${shadow} border w-12 h-12 rounded-full flex items-center justify-center result-badge">
                    ${isPass
                    ? `<svg class="w-7 h-7 ${color}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>`
                    : `<svg class="w-7 h-7 ${color}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12"></path></svg>`
                }
                </div>
            `;
        }
    }
};
