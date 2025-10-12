if (typeof marked !== 'undefined') {
    marked.use({
        gfm: true,  // Enables auto-linking of raw URLs
        breaks: true  // Line breaks as <br>
    });
}


const BASE_URL = window.location.origin;
let chatHistory = [];
// Set PDF.js worker script
if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

function updateFileSize() {
    const fileInput = document.getElementById('compress-file');
    const fileSizeDisplay = document.getElementById('original-file-size');
    if (fileInput && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        fileSizeDisplay.textContent = `Original File Size: ${sizeMB} MB`;
    } else {
        fileSizeDisplay.textContent = 'Original File Size: Not selected';
    }
}


// //
async function computeAllCompressionSizes() {
    console.log('Computing all compression sizes including custom...');
    
    const form = document.getElementById('compressForm');
    const fileInput = form.querySelector('input[type="file"]');
    const resultDiv = document.getElementById('result-compressForm');
    const compressionResults = document.getElementById('compression-results');
    const compressionSizes = document.getElementById('compression-sizes');
    const computeButton = form.querySelector('button[onclick*="computeAllCompressionSizes"]');

    if (!fileInput || !fileInput.files.length) {
        const errorMsg = 'Please select a PDF file first.';
        if (resultDiv) {
            resultDiv.textContent = errorMsg;
            resultDiv.className = 'text-red-600';
        } else {
            alert(errorMsg);
        }
        return;
    }

    const file = fileInput.files[0];
    const originalSizeMB = (file.size / (1024 * 1024)).toFixed(2);

    // Validate file type
    if (file.type !== 'application/pdf') {
        const errorMsg = 'Please select a PDF file.';
        if (resultDiv) {
            resultDiv.textContent = errorMsg;
            resultDiv.className = 'text-red-600';
        }
        return;
    }

    // Disable button during computation
    if (computeButton) {
        computeButton.disabled = true;
        computeButton.innerHTML = '<i class="fas fa-calculator mr-2"></i> Computing...';
    }

    try {
        // Get custom settings if Custom is selected
        const presetSelect = document.getElementById('compress-preset');
        const currentPreset = presetSelect ? presetSelect.value : 'Medium';
        
        // Define all presets including ULTRA Low and custom
      // In computeAllCompressionSizes function - UPDATE THESE:
        const presets = [
            { name: 'High Compression', dpi: 72, quality: 0.6 },      // More aggressive
            { name: 'Medium Compression', dpi: 85, quality: 0.7 },   // Balanced
            { name: 'Low Compression', dpi: 95, quality: 0.8 },      // Light compression
            { name: 'New Compression', dpi: 110, quality: 0.9 }  // 
        ];
        // Add custom preset if Custom is selected
        if (currentPreset === 'Custom') {
            const customDpi = parseInt(document.getElementById('custom_dpi').value) || 100;
            const customQuality = (parseInt(document.getElementById('custom_quality').value) || 70) / 100;
            
            presets.push({
                name: 'Custom Compression',
                dpi: customDpi,
                quality: customQuality
            });
        }

        let sizesHTML = `
            <li class="font-semibold mb-2 text-gray-800">Original Size: ${originalSizeMB} MB</li>
            <hr class="my-2 border-gray-300">
        `;
        
        // Show computing message
        if (resultDiv) {
            resultDiv.innerHTML = '<div class="text-blue-600">🔄 Computing compression sizes... This may take a moment.</div>';
        }

        let computedCount = 0;
        
        for (const preset of presets) {
            console.log(`Testing ${preset.name}...`);
            
            // Update progress in results
            if (compressionSizes) {
                const progress = Math.round((computedCount / presets.length) * 100);
                compressionSizes.innerHTML = sizesHTML + `
                    <li class="text-blue-600">Computing... ${progress}% complete</li>
                    <li class="text-sm text-gray-500">Currently testing: ${preset.name}</li>
                `;
            }
            
            try {
                const compressedBlob = await advancedPDFCompression(
                    file, 
                    preset.dpi, 
                    preset.quality
                );
                
                if (compressedBlob) {
                    const sizeMB = (compressedBlob.size / (1024 * 1024)).toFixed(2);
                    const savings = (((file.size - compressedBlob.size) / file.size) * 100).toFixed(1);
                    
                    const savingsColor = savings >= 50 ? 'text-green-600' : savings >= 20 ? 'text-yellow-600' : 'text-red-600';
                    
                    sizesHTML += `
                        <li class="mb-2 p-3 bg-white rounded-lg border border-gray-200">
                            <strong class="text-gray-800">${preset.name}</strong><br>
                            <span class="text-sm text-gray-600">
                                📏 Size: <strong>${sizeMB} MB</strong> | 
                                💾 Reduction: <strong class="${savingsColor}">${savings}%</strong><br>
                                ⚙️ Settings: ${preset.dpi} DPI, ${Math.round(preset.quality * 100)}% Quality
                            </span>
                        </li>
                    `;
                }
            } catch (error) {
                console.error(`Failed to compute ${preset.name}:`, error);
                sizesHTML += `
                    <li class="mb-2 p-3 bg-red-50 rounded-lg border border-red-200">
                        <strong class="text-red-700">${preset.name}</strong><br>
                        <span class="text-sm text-red-600">❌ Failed to compute size</span>
                    </li>
                `;
            }
            
            computedCount++;
        }

        // Display final results

        // Display final results
if (compressionSizes) {
    compressionSizes.innerHTML = sizesHTML;
}

if (compressionResults) {
    compressionResults.classList.remove('hidden');
}

// 🆕 SIMPLE VERSION: Check for negative savings
if (resultDiv) {
    // Count how many presets have negative savings
    const negativeCount = (sizesHTML.match(/Reduction:.*?-\d+\.?\d*%/g) || []).length;
    const totalPresets = presets.length;
    
    if (negativeCount === totalPresets) {
        // All presets resulted in larger files
        resultDiv.innerHTML = `
            <div class="text-blue-600">
                📊 <strong>PDF Analysis Complete</strong><br>
                <small>This PDF is already highly optimized and cannot be compressed further.</small>
            </div>
        `;
    } else if (negativeCount >= totalPresets / 2) {
        // Majority of presets resulted in larger files
        resultDiv.innerHTML = `
            <div class="text-yellow-600">
                📊 <strong>Limited Compression Potential</strong><br>
                <small>This PDF is already well-compressed. Some presets may increase file size.</small>
            </div>
        `;
    } else {
        // Good compression results
        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Compression Size Estimation Completed!</strong><br>
                <small>Check the table above for size reduction estimates.</small>
            </div>
        `;
    }
}
        // if (compressionSizes) {
        //     compressionSizes.innerHTML = sizesHTML;
        // }
        
        // if (compressionResults) {
        //     compressionResults.classList.remove('hidden');
        // }
        
        // if (resultDiv) {
        //     resultDiv.innerHTML = '<div class="text-green-600">✅ Compression size estimation completed!</div>';
        // }

    } catch (error) {
        console.error('Size computation failed:', error);
        
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="text-red-600">
                    ❌ Failed to compute compression sizes<br>
                    <small>Error: ${error.message}</small>
                </div>
            `;
        }
        
    } finally {
        // Re-enable button
        if (computeButton) {
            computeButton.disabled = false;
            computeButton.innerHTML = '<i class="fas fa-calculator mr-2"></i> Compute All Compression';
        }
    }
}




// // /// //  // NEW ADD SIGNATURE FOR CLIENT SIDE

// Client-side signature addition functions


async function processSignatureClientSide() {
    console.log('Starting client-side signature processing...');
    
    const form = document.getElementById('signatureForm');
    const pdfFileInput = document.getElementById('signature-pdf-file');
    const signatureFileInput = document.getElementById('signature-image-file');
    const selectedPagesInput = document.getElementById('signature-selected-pages');
    const sizeSelect = document.getElementById('signature-size');
    const positionSelect = document.getElementById('signature-position');
    const alignmentSelect = document.getElementById('signature-alignment');
    const removeBgCheckbox = document.getElementById('remove-bg');
    
    // Safely get progress elements with null checks
    const progressDiv = document.getElementById('progress-signatureForm');
    const progressText = document.getElementById('progress-text-signatureForm');
    const resultDiv = document.getElementById('result-signatureForm');
    const submitButton = form ? form.querySelector('button') : null;

    // Validation with better error handling
    if (!pdfFileInput || !pdfFileInput.files[0]) {
        if (resultDiv) {
            resultDiv.textContent = 'Please select a PDF file.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }
    
    if (!signatureFileInput || !signatureFileInput.files[0]) {
        if (resultDiv) {
            resultDiv.textContent = 'Please select a signature image.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }
    
    // Validate signature file type
    const signatureFile = signatureFileInput.files[0];
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    const allowedExtensions = ['.png', '.jpg', '.jpeg'];
    const fileExtension = signatureFile.name.toLowerCase().substring(signatureFile.name.lastIndexOf('.'));
    
    if (!allowedTypes.includes(signatureFile.type) && !allowedExtensions.includes(fileExtension)) {
        if (resultDiv) {
            resultDiv.textContent = 'Please select a PNG or JPEG image file.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }
    
    if (!selectedPagesInput || !selectedPagesInput.value) {
        if (resultDiv) {
            resultDiv.textContent = 'Please select at least one page.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }

    const pdfFile = pdfFileInput.files[0];
    const selectedPages = selectedPagesInput.value.split(',').map(p => parseInt(p.trim()));
    const size = sizeSelect ? sizeSelect.value : 'medium';
    const position = positionSelect ? positionSelect.value : 'bottom';
    const alignment = alignmentSelect ? alignmentSelect.value : 'center';
    const removeBg = removeBgCheckbox ? removeBgCheckbox.checked : false;

    // Validate file sizes
    const pdfSizeMB = pdfFile.size / (1024 * 1024);
    if (pdfSizeMB > 150) {
        if (resultDiv) {
            resultDiv.textContent = 'PDF file exceeds 150MB limit.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }

    const sigSizeMB = signatureFile.size / (1024 * 1024);
    if (sigSizeMB > 10) {
        if (resultDiv) {
            resultDiv.textContent = 'Signature image exceeds 10MB limit.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }

    try {
        // Show progress safely
        if (submitButton) submitButton.disabled = true;
        if (progressDiv) progressDiv.style.display = 'block';
        if (progressText) progressText.textContent = 'Processing signature client-side...';

        console.log('Calling addSignatureClientSide...');
        
        // Process signature
        const signedPdfBlob = await addSignatureClientSide(
            pdfFile, signatureFile, selectedPages, size, position, alignment, removeBg
        );

        if (!signedPdfBlob) {
            throw new Error('Signature addition failed - no blob returned');
        }

        // Download the result
        const url = URL.createObjectURL(signedPdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `signed_${pdfFile.name}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        if (resultDiv) {
            resultDiv.textContent = 'Signature added successfully using client-side processing!';
            resultDiv.classList.remove('text-red-600');
            resultDiv.classList.add('text-green-600');
        }

    } catch (error) {
        console.error('Client-side signature error:', error);
        
        if (resultDiv) {
            resultDiv.textContent = `Error: ${error.message}. Trying server-side fallback...`;
            resultDiv.classList.add('text-red-600');
        }
        
        // Fallback to server-side processing WITHOUT circular call
        await fallbackToServerSideSignature(form);
        
    } finally {
        // Clean up safely
        if (submitButton) submitButton.disabled = false;
        if (progressDiv) progressDiv.style.display = 'none';
        if (progressText) progressText.textContent = '';
    }
}

// Separate fallback function to avoid circular dependency
async function fallbackToServerSideSignature(form) {
    console.log('Falling back to server-side processing...');
    
    // Create a new form data object
    const formData = new FormData();
    
    // Get all form elements safely
    const pdfFileInput = document.getElementById('signature-pdf-file');
    const signatureFileInput = document.getElementById('signature-image-file');
    const selectedPagesInput = document.getElementById('signature-selected-pages');
    const sizeSelect = document.getElementById('signature-size');
    const positionSelect = document.getElementById('signature-position');
    const alignmentSelect = document.getElementById('signature-alignment');
    const removeBgCheckbox = document.getElementById('remove-bg');
    
    // Add files to form data
    if (pdfFileInput && pdfFileInput.files[0]) {
        formData.append('pdf_file', pdfFileInput.files[0]);
    }
    if (signatureFileInput && signatureFileInput.files[0]) {
        formData.append('signature_file', signatureFileInput.files[0]);
    }
    
    // Add other form data
    if (selectedPagesInput && selectedPagesInput.value) {
        formData.append('specific_pages', selectedPagesInput.value);
    }
    if (sizeSelect) {
        formData.append('size', sizeSelect.value);
    }
    if (positionSelect) {
        formData.append('position', positionSelect.value);
    }
    if (alignmentSelect) {
        formData.append('alignment', alignmentSelect.value);
    }
    if (removeBgCheckbox) {
        formData.append('remove_bg', removeBgCheckbox.checked.toString());
    }
    
    try {
        const response = await fetch('/add_signature', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `signed_${pdfFileInput.files[0].name}`;
            a.click();
            window.URL.revokeObjectURL(url);
            
            const resultDiv = document.getElementById('result-signatureForm');
            if (resultDiv) {
                resultDiv.textContent = 'Signature added successfully (server-side fallback)!';
                resultDiv.classList.add('text-green-600');
            }
        } else {
            throw new Error('Server-side processing failed');
        }
    } catch (serverError) {
        console.error('Server-side fallback failed:', serverError);
        const resultDiv = document.getElementById('result-signatureForm');
        if (resultDiv) {
            resultDiv.textContent = `Both client-side and server-side processing failed: ${serverError.message}`;
            resultDiv.classList.add('text-red-600');
        }
    }
}


async function addSignatureClientSide(pdfFile, signatureFile, pages, size, position, alignment, removeBg = false) {
    // Check if PDF-LIB is available
    if (typeof PDFLib === 'undefined') {
        throw new Error('PDF library not loaded. Please refresh the page.');
    }
    
    const { PDFDocument } = PDFLib;
    
    try {
        console.log('Starting client-side signature addition...');
        console.log('Signature file type:', signatureFile.type, 'name:', signatureFile.name);
        
        // Load PDF document
        const pdfBytes = await pdfFile.arrayBuffer();
        const pdfDoc = await PDFDocument.load(pdfBytes);
        
        // Process signature image - support PNG, JPG, JPEG
        let signatureImage;
        const signatureBytes = await signatureFile.arrayBuffer();
        
        if (removeBg) {
            console.log('Removing background client-side...');
            const processedSignature = await removeBackgroundClientSide(signatureFile);
            const processedBytes = await processedSignature.arrayBuffer();
            
            // After background removal, it's always PNG
            signatureImage = await pdfDoc.embedPng(processedBytes);
        } else {
            // Handle PNG, JPG, JPEG based on file type and auto-detection
            const fileType = signatureFile.type.toLowerCase();
            const fileName = signatureFile.name.toLowerCase();
            
            if (fileType === 'image/png' || fileName.endsWith('.png')) {
                console.log('Detected PNG file');
                signatureImage = await pdfDoc.embedPng(signatureBytes);
            } else if (fileType === 'image/jpeg' || fileType === 'image/jpg' || 
                       fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')) {
                console.log('Detected JPEG file');
                signatureImage = await pdfDoc.embedJpg(signatureBytes);
            } else {
                // Auto-detect format by trying both
                console.log('Auto-detecting file format...');
                try {
                    signatureImage = await pdfDoc.embedPng(signatureBytes);
                    console.log('Auto-detected as PNG');
                } catch (pngError) {
                    console.log('Auto-detected as JPEG');
                    signatureImage = await pdfDoc.embedJpg(signatureBytes);
                }
            }
        }
        
        // Size mapping
        const sizeFactors = { 
            small: 0.15, 
            medium: 0.25, 
            large: 0.35 
        };
        const scale = sizeFactors[size] || 0.25;
        
        // Get scaled dimensions
        const { width: originalWidth, height: originalHeight } = signatureImage.scale(1);
        const scaledWidth = originalWidth * scale;
        const scaledHeight = originalHeight * scale;
        
        console.log(`Signature size: ${scaledWidth}x${scaledHeight}, Pages: ${pages}`);
        
        // Add signature to selected pages
        const pageIndices = pages.map(p => p - 1); // Convert to 0-based indexing
        
        for (const pageIndex of pageIndices) {
            if (pageIndex >= 0 && pageIndex < pdfDoc.getPageCount()) {
                const page = pdfDoc.getPage(pageIndex);
                const { width: pageWidth, height: pageHeight } = page.getSize();
                
                // Calculate position
                const coordinates = calculateSignaturePosition(
                    pageWidth, pageHeight, scaledWidth, scaledHeight, position, alignment
                );
                
                console.log(`Adding signature to page ${pageIndex + 1} at position:`, coordinates);
                
                // Draw signature image
                page.drawImage(signatureImage, {
                    x: coordinates.x,
                    y: coordinates.y,
                    width: scaledWidth,
                    height: scaledHeight,
                });
            }
        }
        
        // Save the modified PDF
        const pdfBytesWithSignature = await pdfDoc.save();
        return new Blob([pdfBytesWithSignature], { type: 'application/pdf' });
        
    } catch (error) {
        console.error('Client-side signature addition failed:', error);
        throw new Error(`Client-side processing failed: ${error.message}`);
    }
}


function calculateSignaturePosition(pageWidth, pageHeight, sigWidth, sigHeight, position, alignment) {
    const margin = 50;
    let x, y;
    
    // Vertical position
    switch (position) {
        case 'top':
            y = pageHeight - sigHeight - margin;
            break;
        case 'center':
            y = (pageHeight - sigHeight) / 2;
            break;
        case 'bottom':
        default:
            y = margin;
            break;
    }
    
    // Horizontal alignment
    switch (alignment) {
        case 'left':
            x = margin;
            break;
        case 'center':
            x = (pageWidth - sigWidth) / 2;
            break;
        case 'right':
            x = pageWidth - sigWidth - margin;
            break;
        default:
            x = (pageWidth - sigWidth) / 2;
    }
    
    return { x, y };
}

async function removeBackgroundClientSide(imageFile) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        img.onload = function() {
            try {
                canvas.width = img.width;
                canvas.height = img.height;
                
                // Draw the image
                ctx.drawImage(img, 0, 0);
                
                // Get image data
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;
                
                // Simple white background removal
                for (let i = 0; i < data.length; i += 4) {
                    const r = data[i];
                    const g = data[i + 1];
                    const b = data[i + 2];
                    
                    // Remove white/light background (adjust threshold as needed)
                    if (r > 200 && g > 200 && b > 200) {
                        data[i + 3] = 0; // Set alpha to transparent
                    }
                }
                
                // Put the modified image back
                ctx.putImageData(imageData, 0, 0);
                
                // Convert to PNG blob (always output PNG for transparency)
                canvas.toBlob(blob => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Canvas to blob conversion failed'));
                    }
                }, 'image/png');
                
            } catch (error) {
                reject(error);
            }
        };
        
        img.onerror = function() {
            reject(new Error('Failed to load signature image'));
        };
        
        img.src = URL.createObjectURL(imageFile);
    });
}






// // /// //  // NEW ADD SIGNATURE FOR CLIENT SIDE

// new function to your script.js

async function compressPDFClientSide() {
    console.log('Starting optimized client-side compression...');
    
    const form = document.getElementById('compressForm');
    const fileInput = form.querySelector('input[type="file"]');
    const resultDiv = document.getElementById('result-compressForm');
    const progressDiv = document.getElementById('progress-compressForm');
    const progressText = document.getElementById('progress-text-compressForm');
    const submitButton = form.querySelector('button[type="button"]');

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files.length) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];
    const originalSizeMB = (file.size / (1024 * 1024)).toFixed(2);

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting compression...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Compressing...';

    try {
        // Get compression settings - OPTIMIZED VALUES
        const presetSelect = document.getElementById('compress-preset');
        const preset = presetSelect ? presetSelect.value : 'Medium';
        
        let dpi, quality;
        
        if (preset === 'High') {
            dpi = 72; 
            quality = 0.6;  // Reduced from 0.5 to 0.3
        } else if (preset === 'Medium') {
            dpi = 85; 
            quality = 0.7;  // Reduced from 0.7 to 0.5
        } else if (preset === 'Low') {
            dpi = 95;      // Reduced from 150 to 120
            quality = 0.8;  // Reduced from 0.9 to 0.7
        } else if (preset === 'ULTRA Low') {
            dpi = 110;       // Reduced from 120 to 96
            quality = 0.9;  // Reduced from 0.6 to 0.4
        } else if (preset === 'Custom') {
            const dpiInput = document.getElementById('custom_dpi');
            const qualityInput = document.getElementById('custom_quality');
            
            dpi = dpiInput ? parseInt(dpiInput.value) || 100 : 100;  // Reduced from 150 to 100
            const qualityPercent = qualityInput ? parseInt(qualityInput.value) || 70 : 70;  // Reduced from 80 to 50
            quality = qualityPercent / 100;
        }

        console.log('Using optimized compression settings:', { preset, dpi, quality });

        // Use the advanced compression function
        progressText.textContent = 'Processing PDF pages... (0%)';
        
        let compressedBlob = await advancedPDFCompression(file, dpi, quality, (progress) => {
            progressText.textContent = `Processing pages... (${progress}%)`;
        });

        if (!compressedBlob) {
            throw new Error('Compression returned empty result');
        }

        // 🆕 CHECK: If compression made file LARGER, use aggressive fallback
        const compressedSizeMB = (compressedBlob.size / (1024 * 1024)).toFixed(2);
        let savings = (((file.size - compressedBlob.size) / file.size) * 100).toFixed(1);
        let usedFallback = false;

        if (parseFloat(compressedSizeMB) > parseFloat(originalSizeMB)) {
            console.warn('Compression made file larger, using aggressive fallback...');
            progressText.textContent = 'Optimizing compression...';
            
            // Use more aggressive settings
            const aggressiveDpi = Math.max(72, dpi - 30);
            const aggressiveQuality = Math.max(0.3, quality - 0.2);
            
            const aggressiveBlob = await advancedPDFCompression(file, aggressiveDpi, aggressiveQuality);
            
            if (aggressiveBlob && aggressiveBlob.size < file.size) {
                compressedBlob = aggressiveBlob;
                usedFallback = true;
                
                // Recalculate stats
                const newCompressedSizeMB = (compressedBlob.size / (1024 * 1024)).toFixed(2);
                savings = (((file.size - compressedBlob.size) / file.size) * 100).toFixed(1);
                
                console.log('Fallback compression successful:', {
                    original: originalSizeMB + 'MB',
                    compressed: newCompressedSizeMB + 'MB', 
                    savings: savings + '%',
                    settings: { dpi: aggressiveDpi, quality: aggressiveQuality }
                });
            }
        }

        console.log('Final compression results:', {
            original: originalSizeMB + 'MB',
            compressed: compressedSizeMB + 'MB', 
            savings: savings + '%',
            preset: preset,
            usedFallback: usedFallback
        });

        // Download the compressed file
        const filename = usedFallback 
            ? `compressed_${file.name.replace('.pdf', '')}_optimized.pdf`
            : `compressed_${file.name.replace('.pdf', '')}_${preset.replace(' ', '_')}.pdf`;
        
        if (typeof download === 'function') {
            download(compressedBlob, filename, "application/pdf");
        } else {
            // Fallback download
            const url = window.URL.createObjectURL(compressedBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }

        // Show results with optimization info
        const savingsColor = savings >= 50 ? 'text-green-600' : savings >= 20 ? 'text-yellow-600' : 'text-red-600';
        const fallbackText = usedFallback ? '<br>🔄 <small class="text-blue-600">(Auto-optimized for better compression)</small>' : '';
        
        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Compression Successful!</strong>${fallbackText}<br>
                📁 Original: ${originalSizeMB}MB → Compressed: ${(compressedBlob.size / (1024 * 1024)).toFixed(2)}MB<br>
                💾 Size reduction: <strong class="${savingsColor}">${savings}%</strong><br>
                ⚙️ Preset: <strong>${preset}</strong> (${dpi} DPI, ${Math.round(quality * 100)}% Quality)<br>
                <small class="text-gray-600">(Browser-side processing)</small>
            </div>
        `;

    } catch (error) {
        console.error('Compression failed:', error);
        
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Compression failed: ${error.message}<br>
                <small>Falling back to basic compression...</small>
            </div>
        `;
        
        // Fallback to basic compression
        setTimeout(() => {
            basicPDFCompressionFallback(file, resultDiv, progressText);
        }, 1000);
        
    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Compress PDF';
    }
}

// Enhanced advanced compression function
async function advancedPDFCompression(file, dpi = 100, quality = 0.5, progressCallback) {
    console.log('Starting optimized PDF compression...');
    
    const { PDFDocument } = PDFLib;
    
    try {
        if (progressCallback) progressCallback(10);
        
        // Load the PDF with PDF.js
        const arrayBuffer = await file.arrayBuffer();
        
        if (progressCallback) progressCallback(20);
        
        const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const numPages = pdfDoc.numPages;
        
        const newPdfDoc = await PDFDocument.create();
        
        console.log(`Processing ${numPages} pages with ${dpi} DPI, ${quality * 100}% quality...`);
        
        // 🆕 OPTIMIZATION: Adaptive settings based on file characteristics
        let finalQuality = quality;
        let finalDpi = dpi;
        
        // For very small files, use better quality
        if (file.size < 500000) { // 500KB
            console.log('Small file detected, using better quality...');
            finalQuality = Math.max(quality, 0.7);
        }
        
        // For multi-page documents, be more aggressive
        if (numPages > 10) {
            console.log('Multi-page document, optimizing compression...');
            finalQuality = Math.max(0.3, quality - 0.1);
        }
        
        // For very high DPI, reduce to save space
        if (dpi > 150) {
            console.log('High DPI detected, optimizing...');
            finalDpi = Math.min(150, dpi);
        }

        for (let pageNum = 1; pageNum <= numPages; pageNum++) {
            const progress = 20 + ((pageNum - 1) / numPages) * 70;
            if (progressCallback) progressCallback(Math.round(progress));
            
            const page = await pdfDoc.getPage(pageNum);
            const viewport = page.getViewport({ scale: 1.0 });
            const { width, height } = viewport;
            
            const canvas = document.createElement('canvas');
            const scale = finalDpi / 72;
            canvas.width = width * scale;
            canvas.height = height * scale;
            
            const context = canvas.getContext('2d');
            
            // White background
            context.fillStyle = 'white';
            context.fillRect(0, 0, canvas.width, canvas.height);
            
            // Render PDF page
            const renderContext = {
                canvasContext: context,
                viewport: page.getViewport({ scale: scale }),
            };
            
            await page.render(renderContext).promise;
            
            // Convert to compressed JPEG
            const imageData = canvas.toDataURL('image/jpeg', finalQuality);
            const base64Data = imageData.split(',')[1];
            const imageBytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
            
            // Embed in new PDF
            const image = await newPdfDoc.embedJpg(imageBytes);
            const newPage = newPdfDoc.addPage([width, height]);
            newPage.drawImage(image, {
                x: 0,
                y: 0,
                width: width,
                height: height,
            });
            
            canvas.remove();
        }
        
        if (progressCallback) progressCallback(95);
        
        const compressedBytes = await newPdfDoc.save({
            useObjectStreams: true,
        });
        
        if (progressCallback) progressCallback(100);
        
        return new Blob([compressedBytes], { type: 'application/pdf' });
        
    } catch (error) {
        console.error('Advanced compression failed:', error);
        throw error;
    }
}

// Basic fallback compression
async function basicPDFCompressionFallback(file, resultDiv, progressText) {
    try {
        if (progressText) progressText.textContent = 'Using basic optimization...';
        
        const { PDFDocument } = PDFLib;
        const arrayBuffer = await file.arrayBuffer();
        const pdfDoc = await PDFDocument.load(arrayBuffer);
        
        // Basic optimization - remove metadata, optimize structure
        const pdfBytes = await pdfDoc.save({
            useObjectStreams: true,
            addDefaultPage: false,
        });
        
        const compressedBlob = new Blob([pdfBytes], { type: 'application/pdf' });
        const originalSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        const compressedSizeMB = (compressedBlob.size / (1024 * 1024)).toFixed(2);
        const savings = (((file.size - compressedBlob.size) / file.size) * 100).toFixed(1);
        
        // Download
        const filename = `basic_compressed_${file.name}`;
        download(compressedBlob, filename, "application/pdf");
        
        resultDiv.innerHTML = `
            <div class="text-yellow-600">
                ⚠️ <strong>Basic Compression Applied</strong><br>
                📁 Original: ${originalSizeMB}MB → Compressed: ${compressedSizeMB}MB<br>
                💾 Size reduction: ${savings}%<br>
                <small>(Limited compression - text preserved but smaller size reduction)</small>
            </div>
        `;
        
    } catch (finalError) {
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ All compression methods failed<br>
                <small>Please try server compression instead</small>
            </div>
        `;
    }
}
// 




function initSliders() {
    const dpiSlider = document.getElementById('custom_dpi');
    const qualitySlider = document.getElementById('custom_quality');
    const dpiValue = document.getElementById('dpi-value');
    const qualityValue = document.getElementById('quality-value');

    if (dpiSlider && dpiValue) {
        dpiSlider.addEventListener('input', () => {
            dpiValue.textContent = dpiSlider.value;
        });
    }
    if (qualitySlider && qualityValue) {
        qualitySlider.addEventListener('input', () => {
            qualityValue.textContent = qualitySlider.value;
        });
    }
}

function toggleDeleteInputs() {
    const deleteType = document.getElementById('deletePages-type');
    if (deleteType) {
        const specificPagesInput = document.getElementById('specific-pages-input');
        const rangePagesInput = document.getElementById('range-pages-input');
        if (specificPagesInput && rangePagesInput) {
            specificPagesInput.classList.toggle('hidden', deleteType.value !== 'specific');
            rangePagesInput.classList.toggle('hidden', deleteType.value !== 'range');
        }
    }
}

function toggleCustomInputs() {
    const preset = document.getElementById('compress-preset');
    const customOptions = document.getElementById('custom-compress-options');
    if (preset && customOptions) {
        customOptions.classList.toggle('hidden', preset.value !== 'Custom');
    }
}

function toggleCustomXYInputs() {
    const customPosition = document.querySelector('input[name="position_type"][value="custom"]');
    const customXYInputs = document.getElementById('custom-xy-inputs');
    if (customPosition && customXYInputs) {
        customXYInputs.classList.toggle('hidden', !customPosition.checked);
    }
}

async function getTotalPages(file) {
    if (!file || typeof pdfjsLib === 'undefined') return 0;
    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        return pdf.numPages;
    } catch (err) {
        console.error('Error fetching total pages:', err);
        return 0;
    }
}

async function processPDF(endpoint, formId) {
    console.log(`Processing PDF for endpoint: ${endpoint}, form: ${formId}`);
    const form = document.getElementById(formId);
    const resultDiv = document.getElementById(`result-${formId}`);
    const submitButton = form.querySelector('button');
    const progressDiv = document.getElementById(`progress-${formId}`);
    const progressText = document.getElementById(`progress-text-${formId}`);
    const spinnerStyle = document.createElement('style');

    if (!form) {
        console.error(`Form with ID ${formId} not found`);
        resultDiv.textContent = 'Form not found.';
        resultDiv.classList.add('text-red-600');
        return;
    }

    let formData = new FormData();

    if (endpoint === 'convert_image_to_pdf') {
        const fileInput = form.querySelector('#imageToPdf-file');
        const description = form.querySelector('#image-description')?.value || '';
        const position = form.querySelector('#description-position')?.value || 'top-cente';
        const fontSize = parseInt(form.querySelector('#description-font-size')?.value || '12');
        const pageSize = form.querySelector('select[name="page_size"]')?.value || 'A4';
        const orientation = form.querySelector('select[name="orientation"]')?.value || 'Portrait';
        const customX = form.querySelector('#custom-x')?.value;
        const customY = form.querySelector('#custom-y')?.value;
        const fontColor = form.querySelector('#font-color')?.value || '#000000';
        const fontFamily = form.querySelector('#font-family')?.value || 'helv';
        const fontWeight = form.querySelector('#font-weight')?.value || 'normal';

        if (!fileInput || !fileInput.files[0]) {
            resultDiv.textContent = 'Please select an image file.';
            resultDiv.classList.add('text-red-600');
            return;
        }

        formData.append('file', fileInput.files[0]);
        formData.append('description', description);
        formData.append('description_position', position);
        formData.append('description_font_size', fontSize);
        formData.append('font_color', fontColor); // NEW IMPLEMENTION
        formData.append('font_family', fontFamily); // NEW IMPLEMENTION
        formData.append('font_weight', fontWeight); // NEW IMPLEMENTION
        formData.append('page_size', pageSize);
        formData.append('orientation', orientation);

        if (position === 'custom') {
            if (!customX || !customY || isNaN(customX) || isNaN(customY)) {
                resultDiv.textContent = 'Custom position requires valid X and Y coordinates.';
                resultDiv.classList.add('text-red-600');
                return;
            }
            formData.append('custom_x', parseFloat(customX));
            formData.append('custom_y', parseFloat(customY));
        }
    } else if (endpoint === 'compress_pdf') {
        const preset = form.querySelector('select[name="preset"]').value;
        formData = new FormData(form);
        formData.append('preset', preset);
        if (preset === 'Custom') {
            const customDpi = form.querySelector('input[name="custom_dpi"]').value;
            const customQuality = form.querySelector('input[name="custom_quality"]').value;
            formData.append('custom_dpi', customDpi);
            formData.append('custom_quality', customQuality);
        }
    } else if (endpoint === 'delete_pdf_pages') {
        const deleteType = form.querySelector('select[name="delete_type"]').value;
        const file = form.querySelector('input[type="file"]').files[0];
        let pages;
        if (deleteType === 'specific') {
            pages = form.querySelector('input[name="pages"]').value;
        } else {
            const range = form.querySelector('input[name="range"]').value;
            const totalPages = await getTotalPages(file);
            pages = expandPageRange(range, totalPages);
        }
        formData = new FormData();
        formData.append('file', file);
        formData.append('pages', pages);
    } else {
        const conversionTypeInput = form.querySelector('input[name="conversionType"]:checked');
        formData = new FormData(form);
        if (conversionTypeInput) {
            const conversionType = conversionTypeInput.value;
            formData.append('conversion_type', conversionType);
            console.log('Conversion type:', conversionType);
        }
    }

    if (!(await validateForm(form, endpoint, resultDiv))) {
        console.log('Validation failed');
        return;
    }

    console.log('Sending request to:', `${BASE_URL}/${endpoint}`);
    console.log('FormData contents:');
    for (const [key, value] of formData.entries()) {
        console.log(`${key}: ${value instanceof File ? value.name : value}`);
    }

    submitButton.disabled = true;
    progressDiv.style.display = 'block';
    progressText.textContent = 'Preparing files...';
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + 10, 90);
        progressDiv.querySelector('progress').value = progress;
        progressText.textContent = `Uploading & Processing... ${progress}%`;
    }, 200);

    try {
        const response = await fetch(`${BASE_URL}/${endpoint}`, {
            method: 'POST',
            body: formData
        });
        clearInterval(progressInterval);
        progressDiv.querySelector('progress').value = 100;

        spinnerStyle.textContent = `
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(235, 13, 13, 0.99);
                border-radius: 50%;
                border-top-color: rgba(37, 230, 20, 0.99);
                animation: spin 1s ease-in-out infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(spinnerStyle);
        progressText.innerHTML = 'Processing completed. Wait for download... <i class="fas fa-spinner fa-spin" style="font-size: 1rem;color: red;"></i>';

        if (response.ok) {
            const blob = await response.blob();
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = endpoint === 'compress_pdf' ? 'compressed.pdf' : 'output.pdf';
            if (contentDisposition) {
                const match = contentDisposition.match(/filename="(.+)"|filename=([^;]+)/i);
                if (match) filename = match[1] || match[2];
            }
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            window.URL.revokeObjectURL(url);
            resultDiv.textContent = endpoint === 'compress_pdf' ? 'PDF compressed successfully!' : 'Processing completed successfully!';
            resultDiv.classList.remove('text-red-600');
            resultDiv.classList.add('text-green-600');
        } else {
            const error = await response.json();
            let errorMessage = 'Unknown error';
            if (Array.isArray(error.detail)) {
                errorMessage = error.detail.map(err => `${err.loc.join('.')}: ${err.msg}`).join('; ');
            } else if (error.detail) {
                errorMessage = error.detail;
            }
            console.error('Backend error:', error);
            resultDiv.textContent = `Error: ${errorMessage}`;
            resultDiv.classList.remove('text-green-600');
            resultDiv.classList.add('text-red-600');
        }
    } catch (e) {
        clearInterval(progressInterval);
        console.error('Fetch error:', e, 'Endpoint:', endpoint);
        resultDiv.textContent = `Error: ${e.message}. Please check the server logs.`;
        resultDiv.classList.remove('text-green-600');
        resultDiv.classList.add('text-red-600');
    } finally {
        submitButton.disabled = false;
        setTimeout(() => {
            progressDiv.style.display = 'none';
            progressText.textContent = '';
        }, 2000);
    }
}

function updateFileOrder(files) {
    const fileOrder = Array.from(files).map((file, index) => {
        if (!file.dataset.fileIndex) {
            file.dataset.fileIndex = index;
        }
        return file.dataset.fileIndex;
    });
    const fileOrderInput = document.getElementById('merge-file-order');
    if (fileOrderInput) {
        fileOrderInput.value = fileOrder.join(',');
    }
}


async function validateForm(form, endpoint, resultDiv) {
    const filesInput = form.querySelector('input[type="file"]');
    if (!filesInput) {
        resultDiv.textContent = 'File input not found in the form.';
        resultDiv.classList.add('text-red-600');
        console.error(`No input[type="file"] found in form ${form.id}`);
        return false;
    }
    const files = filesInput.files;
    const password = form.querySelector('input[type="password"]');
    const pages = form.querySelector('input[name="pages"]');
    const pageOrderInput = form.querySelector('input[name="page_order"]');
    const fileOrderInput = form.querySelector('input[name="file_order"]');

    if (!files.length) {
        resultDiv.textContent = 'Please select a file.';
        resultDiv.classList.add('text-red-600');
        return false;
    }

    if (endpoint === 'convert_image_to_pdf') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['image/png', 'image/jpeg'].includes(file.type)) {
            resultDiv.textContent = `File ${file.name} must be PNG or JPEG.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const position = form.querySelector('#description-position')?.value;
        if (!position) {
            resultDiv.textContent = 'Description position field is missing or invalid.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const validPositions = ['top', 'bottom', 'top-left', 'top-center', 'top-right', 
                               'bottom-left', 'bottom-center', 'bottom-right', 'custom'];
        if (!validPositions.includes(position)) {
            resultDiv.textContent = `Invalid description position: ${position}.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (position === 'custom') {
            const customXInput = form.querySelector('#custom-x');
            const customYInput = form.querySelector('#custom-y');
            if (!customXInput || !customYInput) {
                resultDiv.textContent = 'Custom X and Y coordinate inputs are missing.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
            const customX = customXInput.value;
            const customY = customYInput.value;
            if (!customX || !customY || isNaN(customX) || isNaN(customY)) {
                resultDiv.textContent = 'Custom position requires valid X and Y coordinates.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
            const x = parseFloat(customX);
            const y = parseFloat(customY);
            if (x < 0 || y < 0) {
                resultDiv.textContent = 'X and Y coordinates must be non-negative.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
        }
        const fontSize = form.querySelector('#description-font-size')?.value;
        if (!fontSize || isNaN(fontSize) || fontSize < 8 || fontSize > 72) {
            resultDiv.textContent = 'Font size must be a number between 8 and 72.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const pageSize = form.querySelector('select[name="page_size"]')?.value || 'A4';
        if (!['A4', 'Letter'].includes(pageSize)) {
            resultDiv.textContent = `Invalid page size: ${pageSize}. Choose A4 or Letter.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const orientation = form.querySelector('select[name="orientation"]')?.value || 'Portrait';
        if (!['Portrait', 'Landscape'].includes(orientation)) {
            resultDiv.textContent = `Invalid orientation: ${orientation}. Choose Portrait or Landscape.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
    } else if (endpoint === 'add_signature') {
        await processSignatureClientSide();
        return;
        // const pdfFile = files[0];
        // const signatureFile = form.querySelector('input[name="signature_file"]').files[0];
        // const selectedPages = form.querySelector('input[name="specific_pages"]').value;
        // const size = form.querySelector('select[name="size"]').value;
        // const position = form.querySelector('select[name="position"]').value;
        // const alignment = form.querySelector('select[name="alignment"]').value;

        // const pdfSizeMB = pdfFile.size / (1024 * 1024);
        // if (pdfSizeMB > 50) {
        //     resultDiv.textContent = `PDF file ${pdfFile.name} exceeds 50MB limit.`;
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }
        // if (pdfFile.type !== 'application/pdf') {
        //     resultDiv.textContent = `File ${pdfFile.name} must be a PDF.`;
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }

        // const sigSizeMB = signatureFile.size / (1024 * 1024);
        // if (sigSizeMB > 10) {
        //     resultDiv.textContent = `Signature file ${signatureFile.name} exceeds 10MB limit.`;
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }
        // if (!['image/png', 'image/jpeg', 'image/jpg'].includes(signatureFile.type)) {
        //     resultDiv.textContent = `Signature file ${signatureFile.name} must be PNG or JPEG.`;
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }

        // if (!selectedPages) {
        //     resultDiv.textContent = 'Please select at least one page to sign.';
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }
        // if (!/^[1-9]\d*(,[1-9]\d*)*$/.test(selectedPages)) {
        //     resultDiv.textContent = 'Invalid page selection format. Pages must be comma-separated positive integers.';
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }

        // if (!['small', 'medium', 'large'].includes(size)) {
        //     resultDiv.textContent = 'Invalid size. Choose small, medium, or large.';
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }
        // if (!['top', 'center', 'bottom'].includes(position)) {
        //     resultDiv.textContent = 'Invalid position. Choose top, center, or bottom.';
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }
        // if (!['left', 'center', 'right'].includes(alignment)) {
        //     resultDiv.textContent = 'Invalid alignment. Choose left, center, or right.';
        //     resultDiv.classList.add('text-red-600');
        //     return false;
        // }
    } else if (endpoint === 'merge_pdf') {
        if (files.length < 2) {
            resultDiv.textContent = 'Please select at least 2 PDF files.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        let totalSize = 0;
        for (let file of files) {
            const sizeMB = file.size / (1024 * 1024);
            totalSize += sizeMB;
            if (sizeMB > 30) {
                resultDiv.textContent = `File ${file.name} exceeds 30MB limit.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
            if (file.type !== 'application/pdf') {
                resultDiv.textContent = `File ${file.name} must be a PDF.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
        }
        const method = form.querySelector('select[name="method"]').value;
        const maxFiles = method === 'PyPDF2' ? 51 : 30;
        const maxSizeMB = method === 'PyPDF2' ? 90 : 50;
        if (files.length > maxFiles) {
            resultDiv.textContent = `Too many files. Maximum is ${maxFiles} for ${method}.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (totalSize > maxSizeMB) {
            resultDiv.textContent = `Total file size (${totalSize.toFixed(1)}MB) exceeds ${maxSizeMB}MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!fileOrderInput || !fileOrderInput.value) {
            resultDiv.textContent = 'Please load and order files.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const fileOrder = fileOrderInput.value.split(',').map(i => parseInt(i.trim()));
        const uniqueIndices = new Set(fileOrder);
        if (fileOrder.length !== files.length || uniqueIndices.size !== files.length || !fileOrder.every(i => i >= 0 && i < files.length)) {
            resultDiv.textContent = 'Invalid file order. Ensure all files are included exactly once.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
    } else if (endpoint === 'add_page_numbers') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const position = form.querySelector('select[name="position"]').value;
        const alignment = form.querySelector('select[name="alignment"]').value;
        const format = form.querySelector('select[name="format"]').value;
        if (!['top', 'bottom'].includes(position)) {
            resultDiv.textContent = 'Invalid position. Choose top or bottom.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['left', 'center', 'right'].includes(alignment)) {
            resultDiv.textContent = 'Invalid alignment. Choose left, center, or right.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!['page_x', 'x'].includes(format)) {
            resultDiv.textContent = 'Invalid format. Choose Page X or X.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
    } else if (endpoint === 'reorder_pages') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!pageOrderInput || !pageOrderInput.value) {
            resultDiv.textContent = 'Please load and reorder pages.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const totalPages = await getTotalPages(file);
        const pageOrder = pageOrderInput.value.split(',').map(p => parseInt(p.trim()));
        const uniquePages = new Set(pageOrder);
        if (!pageOrder.every(p => /^[1-9]\d*$/.test(p.toString())) || pageOrder.length !== totalPages || uniquePages.size !== totalPages || !pageOrder.every(p => p >= 1 && p <= totalPages)) {
            resultDiv.textContent = `Invalid page order. Must include all pages (1-${totalPages}) exactly once.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
    } else if (endpoint === 'compress_pdf') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 160) {
            resultDiv.textContent = `File ${file.name} exceeds 160MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const preset = form.querySelector('select[name="preset"]').value;
        if (!['High', 'Medium', 'Low', 'Custom'].includes(preset)) {
            resultDiv.textContent = 'Invalid preset. Choose High, Medium, Low, or Custom.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (preset === 'Custom') {
            const customDpi = form.querySelector('input[name="custom_dpi"]').value;
            const customQuality = form.querySelector('input[name="custom_quality"]').value;
            if (!customDpi || !customQuality) {
                resultDiv.textContent = 'Custom preset requires DPI and quality values.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
            const dpi = parseInt(customDpi);
            const quality = parseInt(customQuality);
            if (dpi < 50 || dpi > 400 || quality < 10 || quality > 100) {
                resultDiv.textContent = 'Invalid custom DPI (50-400) or quality (10-100).';
                resultDiv.classList.add('text-red-600');
                return false;
            }
        }
    } else if (endpoint === 'delete_pdf_pages') {
        const deleteTypeElement = form.querySelector('select[name="delete_type"]');
        if (!deleteTypeElement) {
            resultDiv.textContent = 'Delete type selector not found.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const deleteType = deleteTypeElement.value;
        let pagesInput = deleteType === 'specific' ? form.querySelector('input[name="pages"]').value : form.querySelector('input[name="range"]').value;
        console.log('deleteType:', deleteType, 'pagesInput:', pagesInput);
        let totalPages = 0;
        const file = form.querySelector('input[type="file"]').files[0];
        if (!file) {
            resultDiv.textContent = 'Please select a PDF file.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 50) {
            resultDiv.textContent = `File ${file.name} exceeds 50MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (typeof pdfjsLib === 'undefined') {
            resultDiv.textContent = 'PDF processing library not loaded. Please try again later.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        try {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            totalPages = pdf.numPages;
        } catch (err) {
            console.error('PDF page count error:', err);
            resultDiv.textContent = 'Error loading PDF file. Please ensure it’s a valid PDF.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (!pagesInput) {
            resultDiv.textContent = 'Please enter pages to delete.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (deleteType !== 'specific' && deleteType !== 'range') {
            resultDiv.textContent = 'Invalid delete type selected.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
        if (deleteType === 'specific') {
            const pageNumbers = [...new Set(pagesInput.split(',').map(p => p.trim()).filter(p => p))];
            if (!pageNumbers.length || !pageNumbers.every(p => /^[1-9]\d*$/.test(p) && parseInt(p) <= totalPages)) {
                resultDiv.textContent = `Invalid page numbers. Use comma-separated integers (e.g., 1,12,32) within 1-${totalPages}.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
            form.querySelector('input[name="pages"]').value = pageNumbers.join(',');
        } else {
            const expandedPages = expandPageRange(pagesInput, totalPages);
            if (!expandedPages) {
                resultDiv.textContent = `Invalid range. Use format like 1-5 within 1-${totalPages}.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
            form.querySelector('input[name="pages"]').value = expandedPages;
        }
    } else {
        const maxSizeMB = endpoint === 'compress_pdf' ? 55 : endpoint === 'split_pdf' ? 100 : 50;
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > maxSizeMB) {
            resultDiv.textContent = `File ${file.name} exceeds ${maxSizeMB}MB limit.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
    }

    if ((endpoint === 'encrypt_pdf' || endpoint === 'remove_pdf_password') && (!password || !password.value)) {
        resultDiv.textContent = 'Please enter a password.';
        resultDiv.classList.add('text-red-600');
        return false;
    }

    return true;
}

async function displayTotalPages(fileInputId, totalPagesId) {
    const fileInput = document.getElementById(fileInputId);
    const totalPagesDiv = document.getElementById(totalPagesId);
    const resultDiv = document.getElementById(`result-${fileInputId.replace('-file', 'Form')}`);
    if (!fileInput || !fileInput.files[0]) {
        totalPagesDiv.textContent = 'Total Pages: Not loaded';
        return;
    }
    try {
        const file = fileInput.files[0];
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        totalPagesDiv.textContent = `Total Pages: ${pdf.numPages}`;
        resultDiv.textContent = '';
        const pageOrderInput = document.querySelector('input[name="page_order"]');
        if (pageOrderInput && fileInputId === 'reorderPages-file') {
            pageOrderInput.value = Array.from({ length: pdf.numPages }, (_, i) => i + 1).join(',');
        }
    } catch (err) {
        console.error('Error counting pages:', err);
        totalPagesDiv.textContent = 'Total Pages: Error loading PDF';
        resultDiv.textContent = 'Invalid PDF file.';
        resultDiv.classList.add('text-red-600');
    }
}

function expandPageRange(range, totalPages) {
    if (!range || typeof range !== 'string') {
        console.error('Invalid range input:', range);
        return null;
    }
    const trimmedRange = range.trim();
    const match = trimmedRange.match(/^(\d+)-(\d+)$/);
    if (!match) {
        console.error('Range format invalid:', trimmedRange);
        return null;
    }
    const start = parseInt(match[1]);
    const end = parseInt(match[2]);
    if (isNaN(start) || isNaN(end) || start < 1 || end < start || end > totalPages) {
        console.error(`Invalid range values: start=${start}, end=${end}, totalPages=${totalPages}`);
        return null;
    }
    return Array.from({ length: end - start + 1 }, (_, i) => start + i).join(',');
}

function formatResponse(text) {
    if (!text) return '';

    text = text.replace(/(\|[^\n]+\|\r?\n\|[-: |]+\|\r?\n)((?:\|[^\n]+\|\r?\n?)+)/g, function (match, header, body) {
        let html = '<div class="overflow-x-auto"><table class="w-full border-collapse my-3">';
        const headers = header.split('|').slice(1, -1).map(h => h.trim());
        html += '<thead><tr class="bg-gray-100">';
        headers.forEach(h => {
            html += `<th class="p-2 border text-left">${h}</th>`;
        });
        html += '</tr></thead><tbody>';
        const rows = body.trim().split('\n');
        rows.forEach(row => {
            const cells = row.split('|').slice(1, -1).map(c => c.trim());
            html += '<tr>';
            cells.forEach(cell => {
                html += `<td class="p-2 border">${formatMarkdownInline(cell)}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table></div>';
        return html;
    });

    text = text.replace(/^([*-]|\d+\.)\s+(.+)$/gm, function (match, bullet, content) {
        return `<li class="ml-5">${formatMarkdownInline(content)}</li>`;
    });

    text = text.replace(/(<li>.*<\/li>)+/g, function (match) {
        const listType = match.includes('<li class="ml-5">') ? 'ul' : 'ol';
        return `<${listType} class="list-disc pl-5 my-2">${match}</${listType}>`;
    });

    text = formatMarkdownInline(text);

    text = text.replace(/\n/g, function (match, offset, fullText) {
        if (!fullText.substring(offset).match(/^\n*(<table|<ul|<ol)/) &&
            !fullText.substring(0, offset).match(/(<\/table|<\/ul|<\/ol>)\n*$/)) {
            return '<br>';
        }
        return match;
    });

    // UPDATED LINK FORMATTING WITH UNDERLINES
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:underline underline-offset-2 decoration-1" target="_blank" rel="noopener noreferrer">$1</a>');

    // Also handle raw URLs that might not be in markdown format
    text = text.replace(/(https?:\/\/[^\s]+|www\.[^\s]+)/g, '<a href="$1" class="text-blue-600 hover:underline underline-offset-2 decoration-1" target="_blank" rel="noopener noreferrer">$1</a>');

    return text;
}
// Add this CSS
const linkStyles = `
    .chat-link {
        color: #2563eb;
        text-decoration: underline;
        text-underline-offset: 3px;
        text-decoration-thickness: 1.5px;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    .chat-link:hover {
        color: #1d4ed8;
        text-decoration-thickness: 2px;
        background-color: rgba(37, 99, 235, 0.05);
    }
`;

// Add styles to document
if (!document.querySelector('#chat-link-styles')) {
    const styleEl = document.createElement('style');
    styleEl.id = 'chat-link-styles';
    styleEl.textContent = linkStyles;
    document.head.appendChild(styleEl);
}


function formatMarkdownInline(text) {
    if (!text) return '';

    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded">$1</code>');

    return text;
}

async function typewriterEffect(element, text, speed = 20) {
    const formattedText = formatResponse(text);
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = formattedText;
    element.innerHTML = '';
    await typewriterProcessNodes(element, tempDiv.childNodes, speed);
}

async function typewriterProcessNodes(parent, nodes, speed) {
    for (const node of nodes) {
        if (node.nodeType === Node.TEXT_NODE) {
            await typewriterAddText(parent, node.textContent, speed);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            if (node.tagName === 'TABLE') {
                const clone = node.cloneNode(true);
                parent.appendChild(clone);
            } else {
                const newElement = document.createElement(node.tagName);
                for (const attr of node.attributes) {
                    newElement.setAttribute(attr.name, attr.value);
                }
                if (node.className) {
                    newElement.className = node.className;
                }
                parent.appendChild(newElement);
                await typewriterProcessNodes(newElement, node.childNodes, speed);
            }
        }
    }
}

async function typewriterAddText(element, text, speed) {
    for (let i = 0; i < text.length; i++) {
        element.textContent += text[i];
        if (i < text.length - 1) {
            const cursor = document.createElement('span');
            cursor.className = 'blinking-cursor';
            cursor.textContent = '|';
            element.appendChild(cursor);
            await new Promise(resolve => setTimeout(resolve, speed));
            element.removeChild(cursor);
        }
    }
}

// new send chat

async function sendChat() {
    const chatInput = document.getElementById('chatInput');
    const chatOutput = document.getElementById('chatOutput');
    const progressDiv = document.getElementById('progress-chat');
    const progressText = document.getElementById('progress-text-chat');
    const modeToggle = document.getElementById('mode-toggle');
    const modeSelect = document.getElementById('mode-select');

    // if (!chatInput || !chatInput.value.trim()) {
    //     chatOutput.innerHTML = '<p class="text-red-600">Please enter a query.</p>';
    //     return;
    // }

    if (!chatInput || !chatInput.value.trim() || chatInput.value.length > 10000) {
        chatOutput.innerHTML = `<p class="text-red-600">${!chatInput.value.trim() ? 'Please enter a query.' : 'Query too long (max 10000 characters)'}</p>`;
        return;
    }

    const userMessage = chatInput.value.trim();
    
    console.log("📝 Current chatHistory BEFORE adding new message:", chatHistory);

    progressDiv.style.display = 'block';
    progressText.textContent = 'Processing query...';

    try {
        // Get the selected mode
        let selectedMode = null;
        if (modeToggle && modeToggle.checked && modeSelect && modeSelect.value) {
            selectedMode = modeSelect.value;
        }
        
        // ✅ FIX: Send only LAST 4 conversations (8 messages) to LLM
        const recentHistory = chatHistory.slice(-8); // Last 4 conversations (4 user + 4 assistant)
        
        const body = new URLSearchParams({
            query: userMessage,
            mode: selectedMode || '',
            history: JSON.stringify(recentHistory) // Send only recent history
        });

        console.log("📤 Sending to backend - Query:", userMessage);
        console.log("📤 Sending to backend - Recent History (last 4 convos):", JSON.stringify(recentHistory));
        console.log("📊 Full history length:", chatHistory.length, "Recent history length:", recentHistory.length);

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: body
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Chat error');
        }

        const data = await response.json();

        // ✅ Add BOTH user message and AI response to FULL history (for UI)
        chatHistory.push({ role: 'user', content: userMessage });
        chatHistory.push({ role: 'assistant', content: data.answer });

        // Update UI with ALL messages
        const messageDiv = document.createElement('div');
        messageDiv.className = 'mb-4';

        const userQuery = document.createElement('p');
        userQuery.className = 'font-semibold text-blue-600';
        userQuery.textContent = `You: ${userMessage}`;
        messageDiv.appendChild(userQuery);

        const aiResponse = document.createElement('div');
        aiResponse.className = 'ai-response bg-gray-50 p-3 rounded mt-1';
        messageDiv.appendChild(aiResponse);

        chatOutput.insertBefore(messageDiv, chatOutput.firstChild);

        chatInput.value = '';

        // Use marked.parse for proper markdown rendering
        if (typeof marked !== 'undefined') {
            aiResponse.innerHTML = marked.parse(data.answer);
        } else {
            aiResponse.textContent = data.answer;
        }

        // Keep full history for UI (optional: limit to prevent memory issues)
        if (chatHistory.length > 100) { // Keep reasonable limit for browser memory
            chatHistory = chatHistory.slice(-100);
            console.log("🗂️ Trimmed full history to 100 messages for UI");
        }

        // DEBUG: Log after updating history
        console.log("📝 Updated FULL chatHistory (UI):", chatHistory.length, "messages");
        console.log("📝 Next LLM will receive:", Math.min(chatHistory.length, 8), "messages");

    } catch (error) {
        console.error("Chat error:", error);
        // Don't add failed message to history
        const errorDiv = document.createElement('div');
        errorDiv.className = 'text-red-600 mb-4';
        errorDiv.innerHTML = `
            <p>Error: ${error.message}</p>
            <p class="text-sm text-gray-600">Please try again or refresh the page.</p>
        `;
        chatOutput.insertBefore(errorDiv, chatOutput.firstChild);
    } finally {
        progressDiv.style.display = 'none';
        progressText.textContent = '';
    }
}



function showTool(toolId) {
    localStorage.setItem('lastTool', toolId);
    console.log("checking toolid ",toolId);
    
    document.querySelectorAll('.tool-section').forEach(section => {
        section.style.display = 'none';
    });
    const toolSection = document.getElementById(toolId);
    if (toolSection) {
        toolSection.style.display = 'block';
    }
    // document.getElementById('chat-section').style.display = 'block';

    // hide and show clear form
    const clearBtn = document.getElementById('clear-all-btn-container');
    if (toolId === 'chat-section') {
      clearBtn.style.display = 'none';
    } else {
      clearBtn.style.display = 'block';
    }

    // 
    document.querySelectorAll('.nav-link, .dropdown-content a').forEach(link => {
        link.classList.remove('text-green-600');
        link.classList.add('text-blue-600');
    });

    if (event && event.currentTarget) {
        event.currentTarget.classList.add('text-green-600');
    }

    // Updated mobile dropdown handling
    const mobileMenu = document.querySelector('#mobile-menu');
    const mobileSubmenu = document.querySelector('#mobile-submenu');
    const menuButton = document.querySelector('#mobile-menu-button');
    if (mobileMenu && window.innerWidth <= 768) {
        mobileMenu.classList.add('hidden');
        if (mobileSubmenu) {
            mobileSubmenu.classList.add('hidden');
        }
        if (menuButton) {
            menuButton.querySelector('i').classList.remove('fa-times');
            menuButton.querySelector('i').classList.add('fa-bars');
        }
    }
}

async function processImage(endpoint, formId) {
    const form = document.getElementById(formId);
    const resultDiv = document.getElementById(`result-${formId}`);
    const progress = document.getElementById(`progress-${formId}`);
    const progressText = document.getElementById(`progress-text-${formId}`);
    const fileInput = form.querySelector('input[type="file"]');

    if (!fileInput || !fileInput.files[0]) {
        resultDiv.textContent = 'Please select an image file.';
        resultDiv.classList.add('text-red-600');
        return;
    }

    const formData = new FormData(form);
    progress.style.display = 'block';
    progressText.textContent = 'Processing...';
    form.querySelector('button').disabled = true;

    try {
        const response = await fetch(`/${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'processed_image.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        resultDiv.textContent = 'Background removed successfully! Image downloaded.';
        resultDiv.classList.remove('text-red-600');
        resultDiv.classList.add('text-green-600');
    } catch (err) {
        resultDiv.textContent = `Error: ${err.message}`;
        resultDiv.classList.add('text-red-600');
    } finally {
        progress.style.display = 'none';
        progressText.textContent = '';
        form.querySelector('button').disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initSliders();
    updateFileSize();
    const deletePagesType = document.getElementById('deletePages-type');
    if (deletePagesType) {
        deletePagesType.addEventListener('change', toggleDeleteInputs);
    }

 const modeToggle = document.getElementById('mode-toggle');
    const modeSelect = document.getElementById('mode-select');
    const chatInput = document.getElementById('chatInput');
    const modeLabel = document.querySelector('.ms-3.text-sm.font-medium.text-gray-900');
    
    if (modeToggle && modeSelect && chatInput && modeLabel) {
        // Function to update UI based on toggle state
        function updateModeUI() {
          
            if (modeToggle.checked) {
                // Tone Selector ENABLED
                chatInput.placeholder = "Ask anything (except Vishnu)...";
                modeLabel.innerHTML = 'Tone Selector <br> <span class="text-xs text-gray-500">General Mode</span>';
                modeSelect.style.display = 'flex';
            } else {
                // Tone Selector DISABLED
                chatInput.placeholder = "Ask anything about Vishnu...";
                modeLabel.innerHTML = 'Tone Selector <br> <span class="text-xs text-gray-500">Vishnu Mode</span>';
                modeSelect.style.display = 'none';
            }
        }
        
        // Initial setup
        updateModeUI();
        
        // Update on toggle change
        modeToggle.addEventListener('change', function() {
            modeSelect.disabled = !this.checked;
            if (this.checked && !modeSelect.value) {
                modeSelect.value = "general"; // Auto-select General mode
            }
            updateModeUI();
        });
        
        // Also update when mode selection changes
        modeSelect.addEventListener('change', updateModeUI);
    }






    // Initialize merge_pdf file reordering
    const mergeFileInput = document.getElementById('mergePdf-file');
    const fileList = document.querySelector('.file-list');
    if (mergeFileInput && fileList) {
        mergeFileInput.addEventListener('change', () => {
            fileList.innerHTML = '';
            const files = mergeFileInput.files;
            Array.from(files).forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.dataset.fileIndex = index;
                fileItem.className = 'file-item flex justify-between items-center mb-2';
                fileItem.innerHTML = `
                    <span>${file.name}</span>
                    <div>
                        <button class="move-up bg-blue-600 text-white px-2 py-1 rounded mr-2">Up</button>
                        <button class="move-down bg-blue-600 text-white px-2 py-1 rounded">Down</button>
                    </div>
                `;
                fileList.appendChild(fileItem);
            });
            updateFileOrder(fileList.querySelectorAll('.file-item'));
            updateButtonStates(fileList.querySelectorAll('.file-item'));
        });

        // Handle move up/down buttons for merge_pdf
        fileList.addEventListener('click', (e) => {
            const target = e.target;
            if (target.classList.contains('move-up') || target.classList.contains('move-down')) {
                const fileItem = target.closest('.file-item');
                const items = Array.from(fileList.querySelectorAll('.file-item'));
                const index = items.indexOf(fileItem);
                if (target.classList.contains('move-up') && index > 0) {
                    fileList.insertBefore(fileItem, items[index - 1]);
                } else if (target.classList.contains('move-down') && index < items.length - 1) {
                    fileList.insertBefore(items[index + 1], fileItem);
                }
                updateFileOrder(fileList.querySelectorAll('.file-item'));
                // updateButtonStates(fileList.querySelectorAll('.file-item'));
            }
        });
    }

    // Initialize reorder_pages
    const reorderFileInput = document.getElementById('reorderPages-file');
    const pageList = document.querySelector('.page-list');
    if (reorderFileInput && pageList) {
        reorderFileInput.addEventListener('change', async () => {
            await displayTotalPages('reorderPages-file', 'total-pages-reorderPages');
            const pageOrderInput = document.querySelector('input[name="page_order"]');
            if (pageOrderInput && pageOrderInput.value) {
                pageList.innerHTML = '';
                const pages = pageOrderInput.value.split(',').map(p => parseInt(p.trim()));
                pages.forEach((page, index) => {
                    const pageItem = document.createElement('div');
                    pageItem.dataset.fileIndex = index;
                    pageItem.dataset.pageNumber = page;
                    pageItem.className = 'page-item flex justify-between items-center mb-2';
                    pageItem.innerHTML = `
                        <span>Page ${page}</span>
                        <div>
                            <button class="move-up bg-blue-600 text-white px-2 py-1 rounded mr-2">Up</button>
                            <button class="move-down bg-blue-600 text-white px-2 py-1 rounded">Down</button>
                        </div>
                    `;
                    pageList.appendChild(pageItem);
                });
                updateButtonStates(pageList.querySelectorAll('.page-item'));

                // Handle page reordering
                pageList.addEventListener('click', (e) => {
                    const target = e.target;
                    if (target.classList.contains('move-up') || target.classList.contains('move-down')) {
                        const pageItem = target.closest('.page-item');
                        const items = Array.from(pageList.querySelectorAll('.page-item'));
                        const index = items.indexOf(pageItem);
                        if (target.classList.contains('move-up') && index > 0) {
                            pageList.insertBefore(pageItem, items[index - 1]);
                        } else if (target.classList.contains('move-down') && index < items.length - 1) {
                            pageList.insertBefore(items[index + 1], pageItem);
                        }
                        const newOrder = Array.from(pageList.querySelectorAll('.page-item')).map(item => item.dataset.pageNumber);
                        pageOrderInput.value = newOrder.join(',');
                        // updateButtonStates(pageList.querySelectorAll('.page-item'));
                    }
                }, { once: true }); // Prevent duplicate listeners
            }
        });
    }

    // Initialize image to PDF functionality
    const imageToPdfForm = document.getElementById('imageToPdfForm');
    if (imageToPdfForm) {
        const imageToPdfFile = document.getElementById('imageToPdf-file');
        const descriptionPosition = document.getElementById('description-position');
        if (imageToPdfFile) {
            imageToPdfFile.addEventListener('change', function() {
                const fileName = this.files[0] ? this.files[0].name : 'No file selected';
                const fileNameElement = document.getElementById('imageToPdf-file-name');
                if (fileNameElement) {
                    fileNameElement.textContent = fileName;
                } else {
                    console.warn('Element with ID "imageToPdf-file-name" not found');
                }
            });
        } else {
            console.warn('Element with ID "imageToPdf-file" not found');
        }

        if (descriptionPosition) {
            descriptionPosition.addEventListener('change', function() {
                const customContainer = document.getElementById('custom-position-container');
                const customX = document.getElementById('custom-x');
                const customY = document.getElementById('custom-y');
                if (customContainer) {
                    const isCustom = this.value === 'custom';
                    customContainer.classList.toggle('hidden', !isCustom);
                    if (!isCustom && customX && customY) {
                        customX.value = '';
                        customY.value = '';
                    }
                } else {
                    console.warn('Element with ID "custom-position-container" not found');
                }
            });
            // Trigger change event on load
            descriptionPosition.dispatchEvent(new Event('change'));
        } else {
            console.warn('Element with ID "description-position" not found');
        }
    } else {
        console.warn('Form with ID "imageToPdfForm" not found');
    }
});





function clearAllForms() {
    // Reset all forms on the page
    document.querySelectorAll('form').forEach(form => form.reset());
    
    // Clear all file inputs
    document.querySelectorAll('input[type="file"]').forEach(input => {
      input.value = '';
      // Trigger change event to update UI
      input.dispatchEvent(new Event('change'));
    });
    
    // Reset all file name displays
    document.querySelectorAll('[id$="-file-name"], [id$="-files-count"]').forEach(display => {
      display.textContent = 'No file selected';
    });
    
    // Clear all result messages
    document.querySelectorAll('[id^="result-"]').forEach(result => {
      result.textContent = '';
    });
    

  }

  


