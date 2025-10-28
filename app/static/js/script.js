if (typeof marked !== 'undefined') {
    marked.use({
        gfm: true,  // Enables auto-linking of raw URLs
        breaks: true  // Line breaks as <br>
    });
}


const BASE_URL = window.location.origin;
let chatHistory = [];
const MAX_FILE_SIZE_PDFWORDEXCEL = 5 * 1024 * 1024; // 10MB
const PDFWORDEXCEL_MAX_PAGES = 5;
////////////////////////////////////////////////////ANALYZE PDF 

// Analyze PDF to determine content type
async function analyzePDFContent(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdfjs = pdfLibraryManager.libraries.pdfjs.lib;
    const pdfDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;

    let textCount = 0;
    let imageCount = 0;
    let totalPages = pdfDoc.numPages;
    let hasComplexImages = false;

    // Sample first few pages for analysis
    // const samplePages = Math.min(5, totalPages);

    // for (let i = 1; i <= samplePages; i++) {
    for (let i = 1; i <= totalPages; i++) {
        const page = await pdfDoc.getPage(i);
        const ops = await page.getOperatorList();

        // Analyze operators to detect content
        const textOps = ops.fnArray.filter(op =>
            op === pdfjs.OPS.showText ||
            op === pdfjs.OPS.showTextGlyphPositioning
        ).length;

        const imageOps = ops.fnArray.filter(op =>
            op === pdfjs.OPS.paintImageXObject ||
            op === pdfjs.OPS.paintJpegXObject
        ).length;

        textCount += textOps;
        imageCount += imageOps;

        // Check for high-resolution images
        if (imageOps > 0) {
            const viewport = page.getViewport({ scale: 1.0 });
            if (viewport.width > 1000 || viewport.height > 1000) {
                hasComplexImages = true;
            }
        }
    }

    // Determine content type
    const textRatio = textCount / (textCount + imageCount);
    const isTextHeavy = textRatio > 0.7;
    const isImageHeavy = textRatio < 0.3;

    let compressionStrategy, contentType;

    if (isTextHeavy && !hasComplexImages) {
        compressionStrategy = "Text Optimization";
        contentType = "Text-heavy document";
    } else if (isImageHeavy) {
        compressionStrategy = "Smart Image Compression";
        contentType = "Image-heavy document";
    } else {
        compressionStrategy = "Balanced Compression";
        contentType = "Mixed content document";
    }

    return {
        totalPages,
        textCount,
        imageCount,
        textRatio,
        isTextHeavy,
        isImageHeavy,
        hasComplexImages,
        compressionStrategy,
        contentType
    };
}



//////////////////////////////////////////////////////////////////////////////////////////////////
// async function compressPDFClientSide() {
//     console.log('Starting intelligent PDF compression...');

//     const [pdfjs, pdfLib] = await pdfLibraryManager.loadLibraries(['pdfjs', 'pdfLib']);
//     const { PDFDocument } = pdfLib;

//     const form = document.getElementById('compressForm');
//     const fileInput = form.querySelector('input[type="file"]');
//     const resultDiv = document.getElementById('result-compressForm');
//     const progressDiv = document.getElementById('progress-compressForm');
//     const progressText = document.getElementById('progress-text-compressForm');
//     const submitButton = form.querySelector('button[type="button"]');

//     if (!fileInput?.files?.length) {
//         alert('Please select a PDF file.');
//         return;
//     }

//     const file = fileInput.files[0];
//     const originalSizeMB = (file.size / (1024 * 1024)).toFixed(2);

//     // Show progress
//     progressDiv.style.display = 'block';
//     progressText.textContent = 'Analyzing PDF content...';
//     submitButton.disabled = true;
//     submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Analyzing...';

//     try {
//         // Step 1: Analyze PDF content type
//         progressText.textContent = 'Analyzing PDF structure...';
//         const pdfAnalysis = await analyzePDFContent(file);

//         console.log('PDF Analysis:', pdfAnalysis);

//         // Step 2: Choose compression strategy
//         let compressedBlob;
//         if (pdfAnalysis.isTextHeavy && !pdfAnalysis.hasComplexImages) {
//             compressedBlob = await optimizeTextPDF(file, pdfAnalysis);
//         } else if (pdfAnalysis.isImageHeavy) {
//             compressedBlob = await compressImagePDF(file, pdfAnalysis);
//         } else {
//             compressedBlob = await compressMixedPDF(file, pdfAnalysis);
//         }

//         if (!compressedBlob) {
//             throw new Error('Compression failed');
//         }

//         // Calculate results
//         const compressedSizeMB = (compressedBlob.size / (1024 * 1024)).toFixed(2);
//         const savings = (((file.size - compressedBlob.size) / file.size) * 100).toFixed(1);

//         // Only download if actual compression occurred
//         if (parseFloat(compressedSizeMB) < parseFloat(originalSizeMB)) {
//             const filename = `compressed_${file.name.replace('.pdf', '')}.pdf`;
//             downloadBlob(compressedBlob, filename);
//         }

//         // Show results
//         resultDiv.innerHTML = `
//             <div class="${parseFloat(savings) > 0 ? 'text-green-600' : 'text-red-600'}">
//                 ${parseFloat(savings) > 0 ? '✅' : '⚠️'} 
//                 <strong>${parseFloat(savings) > 0 ? 'Compression Successful!' : 'Compression Ineffective'}</strong><br>
//                 📁 Original: ${originalSizeMB}MB → Compressed: ${compressedSizeMB}MB<br>
//                 💾 Size reduction: <strong class="${savings >= 0 ? 'text-green-600' : 'text-red-600'}">${savings}%</strong><br>
//                 🎯 Strategy: ${pdfAnalysis.compressionStrategy}<br>
//                 📊 Content: ${pdfAnalysis.contentType}
//                 ${parseFloat(savings) <= 0 ? '<br><small class="text-orange-600">This PDF is already optimized. Try a different compression strategy.</small>' : ''}
//             </div>
//         `;

//     } catch (error) {
//         console.error('Compression failed:', error);
//         resultDiv.innerHTML = `
//             <div class="text-red-600">
//                 ❌ Compression failed: ${error.message}
//             </div>
//         `;
//     } finally {
//         progressDiv.style.display = 'none';
//         submitButton.disabled = false;
//         submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Compress PDF';
//     }
// }

// FIXED: Optimize text-heavy PDFs WITHOUT converting to images
async function optimizeTextPDF(file, analysis) {
    const arrayBuffer = await file.arrayBuffer();
    const pdfLib = pdfLibraryManager.libraries.pdfLib.lib;
    const { PDFDocument } = pdfLib;

    try {
        // Method 1: Direct optimization (preserves vectors)
        const pdfDoc = await PDFDocument.load(arrayBuffer);

        // Remove unnecessary metadata and optimize structure
        const compressedBytes = await pdfDoc.save({
            useObjectStreams: true,
            objectsPerStream: 50,
            // Only compress images, preserve text as vectors
            compressImages: true,
            imageQuality: 0.7,
            // Remove unused objects
            removeUnusedObjects: true,
            // Optimize for web
            addDefaultPage: false
        });

        const optimizedBlob = new Blob([compressedBytes], { type: 'application/pdf' });

        // Only use image conversion if it actually reduces size
        if (optimizedBlob.size < file.size) {
            return optimizedBlob;
        } else {
            // Fallback: Try with more aggressive image compression
            const pdfDoc2 = await PDFDocument.load(arrayBuffer);
            const compressedBytes2 = await pdfDoc2.save({
                useObjectStreams: true,
                compressImages: true,
                imageQuality: 0.5, // More aggressive
                removeUnusedObjects: true
            });

            return new Blob([compressedBytes2], { type: 'application/pdf' });
        }

    } catch (error) {
        console.warn('Direct optimization failed, using fallback:', error);
        // Fallback to minimal image conversion
        return await compressWithMinimalRasterization(file, analysis, 2.0, 0.9);
    }
}

// FIXED: Smart compression for image-heavy PDFs
async function compressImagePDF(file, analysis) {
    const arrayBuffer = await file.arrayBuffer();
    const pdfjs = pdfLibraryManager.libraries.pdfjs.lib;
    const pdfLib = pdfLibraryManager.libraries.pdfLib.lib;
    const { PDFDocument } = pdfLib;

    try {
        // First try direct optimization
        const pdfDoc = await PDFDocument.load(arrayBuffer);
        const compressedBytes = await pdfDoc.save({
            useObjectStreams: true,
            compressImages: true,
            imageQuality: 0.6, // Aggressive for images
            removeUnusedObjects: true
        });

        const directOptimized = new Blob([compressedBytes], { type: 'application/pdf' });

        // If direct optimization works, use it
        if (directOptimized.size < file.size * 0.9) { // At least 10% reduction
            return directOptimized;
        }
    } catch (error) {
        console.warn('Direct image optimization failed:', error);
    }

    // Fallback to selective rasterization
    return await compressWithSelectiveRasterization(file, analysis, 1.5, 0.8);
}

// NEW: Minimal rasterization - only when necessary
async function compressWithMinimalRasterization(file, analysis, scale, quality) {
    const arrayBuffer = await file.arrayBuffer();
    const pdfjs = pdfLibraryManager.libraries.pdfjs.lib;
    const pdfLib = pdfLibraryManager.libraries.pdfLib.lib;
    const { PDFDocument } = pdfLib;

    const sourceDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
    const newPdfDoc = await PDFDocument.create();

    const totalPages = sourceDoc.numPages;

    for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
        const page = await sourceDoc.getPage(pageNum);
        const viewport = page.getViewport({ scale: 1.0 });

        // Check if page has complex elements that need rasterization
        const needsRasterization = await shouldRasterizePage(page, analysis);

        if (needsRasterization) {
            await compressPageAsImageHighQuality(page, newPdfDoc, viewport, scale, quality);
        } else {
            // Try to copy page directly (preserve vectors)
            try {
                await copyPageDirectly(page, newPdfDoc, viewport);
            } catch (error) {
                // Fallback to high-quality rasterization
                await compressPageAsImageHighQuality(page, newPdfDoc, viewport, 2.0, 0.95);
            }
        }
    }

    const compressedBytes = await newPdfDoc.save({
        useObjectStreams: true,
        compressImages: true,
        imageQuality: 0.8
    });

    return new Blob([compressedBytes], { type: 'application/pdf' });
}

// NEW: Determine if page should be rasterized
async function shouldRasterizePage(page, analysis) {
    try {
        const ops = await page.getOperatorList();

        // Count different types of operations
        const imageOps = ops.fnArray.filter(op =>
            op === pdfjs.OPS.paintImageXObject ||
            op === pdfjs.OPS.paintJpegXObject
        ).length;

        const textOps = ops.fnArray.filter(op =>
            op === pdfjs.OPS.showText ||
            op === pdfjs.OPS.showTextGlyphPositioning
        ).length;

        const pathOps = ops.fnArray.filter(op =>
            op === pdfjs.OPS.fill ||
            op === pdfjs.OPS.stroke
        ).length;

        // Rasterize if:
        // - Many images
        // - Complex paths (charts, diagrams)
        // - Few text operations (not text-heavy)
        const shouldRasterize =
            imageOps > 3 ||
            (pathOps > 20 && textOps < 10) ||
            (analysis.hasComplexImages && imageOps > 0);

        return shouldRasterize;
    } catch (error) {
        console.warn('Page analysis failed, defaulting to rasterization:', error);
        return true;
    }
}

// NEW: High-quality page compression
async function compressPageAsImageHighQuality(page, newPdfDoc, viewport, scale, quality) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d', {
        alpha: false,
        imageSmoothingEnabled: true,
        imageSmoothingQuality: 'high'
    });

    // Calculate optimal DPI
    const optimalScale = calculateOptimalDPIScale(viewport, scale);
    canvas.width = Math.floor(viewport.width * optimalScale);
    canvas.height = Math.floor(viewport.height * optimalScale);

    // High-quality rendering settings
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';

    const renderContext = {
        canvasContext: context,
        viewport: page.getViewport({ scale: optimalScale }),
        enableWebGL: true,
        renderInteractiveForms: false
    };

    await page.render(renderContext).promise;

    // Use PNG for better quality, JPEG only for very large images
    const hasText = await detectTextOnPage(page);
    const usePNG = hasText || quality > 0.85;

    const imageData = usePNG ?
        canvas.toDataURL('image/png') :
        canvas.toDataURL('image/jpeg', Math.max(0.7, quality));

    const base64Data = imageData.split(',')[1];
    const imageBytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));

    const image = usePNG ?
        await newPdfDoc.embedPng(imageBytes) :
        await newPdfDoc.embedJpg(imageBytes);

    const newPage = newPdfDoc.addPage([viewport.width, viewport.height]);
    newPage.drawImage(image, {
        x: 0, y: 0,
        width: viewport.width,
        height: viewport.height,
    });

    canvas.remove();
}

// NEW: Try to copy page directly (preserve vectors)
async function copyPageDirectly(page, newPdfDoc, viewport) {
    // This is a simplified version - in practice, you'd need more complex PDF manipulation
    // For now, we'll use high-quality rasterization as fallback
    throw new Error('Direct copy not implemented, using high-quality fallback');
}

// NEW: Smart DPI calculation
function calculateOptimalDPIScale(viewport, baseScale) {
    const baseWidth = viewport.width;
    const baseHeight = viewport.height;
    const maxDimension = Math.max(baseWidth, baseHeight);

    // Higher DPI for smaller documents, lower for larger ones
    if (maxDimension > 1500) return baseScale * 0.8;  // Large docs: lower DPI
    if (maxDimension > 1000) return baseScale * 1.0;  // Medium docs
    if (maxDimension > 500) return baseScale * 1.2;   // Small docs
    return baseScale * 1.5;                           // Very small: high DPI
}

// FIXED: Mixed content compression
async function compressMixedPDF(file, analysis) {
    // Try direct optimization first
    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdfLib = pdfLibraryManager.libraries.pdfLib.lib;
        const { PDFDocument } = pdfLib;

        const pdfDoc = await PDFDocument.load(arrayBuffer);
        const compressedBytes = await pdfDoc.save({
            useObjectStreams: true,
            compressImages: true,
            imageQuality: 0.7,
            removeUnusedObjects: true
        });

        const optimized = new Blob([compressedBytes], { type: 'application/pdf' });

        if (optimized.size < file.size) {
            return optimized;
        }
    } catch (error) {
        console.warn('Direct mixed optimization failed:', error);
    }

    // Fallback to selective rasterization
    return await compressWithSelectiveRasterization(file, analysis, 1.5, 0.8);
}

// NEW: Selective rasterization
async function compressWithSelectiveRasterization(file, analysis, scale, quality) {
    const arrayBuffer = await file.arrayBuffer();
    const pdfjs = pdfLibraryManager.libraries.pdfjs.lib;
    const pdfLib = pdfLibraryManager.libraries.pdfLib.lib;
    const { PDFDocument } = pdfLib;

    const sourceDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
    const newPdfDoc = await PDFDocument.create();

    const totalPages = sourceDoc.numPages;

    for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
        const page = await sourceDoc.getPage(pageNum);
        const viewport = page.getViewport({ scale: 1.0 });

        const needsRasterization = await shouldRasterizePage(page, analysis);

        if (needsRasterization) {
            await compressPageAsImageHighQuality(page, newPdfDoc, viewport, scale, quality);
        } else {
            // For text-heavy pages, use higher quality
            await compressPageAsImageHighQuality(page, newPdfDoc, viewport, 2.0, 0.9);
        }
    }

    const compressedBytes = await newPdfDoc.save();
    return new Blob([compressedBytes], { type: 'application/pdf' });
}

// Utility function for download
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Helper function to detect text on page
async function detectTextOnPage(page) {
    try {
        const textContent = await page.getTextContent();
        return textContent.items.length > 2;
    } catch (error) {
        return false;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// Replace the existing computeAllCompressionSizes function
async function computeAllCompressionSizes() {
    const compressionType = document.querySelector('input[name="compression_type"]:checked');
    const compbutton = document.getElementById('compress-submit-btn');
    // compbutton.disabled = true;

    try {
        if (compressionType && compressionType.value === 'server') {

            
        
            await computeServerCompressionSizesTwoStep();
        } else {
            await computeClientCompressionSizes();
        }
    } catch (error) {
        console.error('Size computation failed:', error);
    } finally {
        // ✅ ALWAYS re-enable the button, even if there's an error
        compbutton.disabled = false;
    }
}
/////////////////////////////////////////////////////
let currentTaskId = null;
let progressInterval = null;


function stopProgressTracking() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    currentTaskId = null;
}



function startProgressTracking(taskId, progressBar, progressText, progressPercent, progressStatus) {
    console.log('🚀 Starting progress tracking for task:', taskId);

    if (!taskId) {
        console.error('❌ No taskId provided');
        return null;
    }

    let isCompleted = false;

    const progressInterval = setInterval(async () => {
        if (isCompleted) {
            clearInterval(progressInterval);
            return;
        }

        try {
            const response = await fetch(`${BASE_URL}/progress/${taskId}`);

            if (response.ok) {
                const progressData = await response.json();
                console.log('📊 Progress update received:', progressData);

                // ✅ ALWAYS update UI with received data
                if (progressData.progress !== undefined) {
                    updateProgressUI(progressBar, progressText, progressPercent, progressStatus, progressData);

                    // ✅ Check for completion
                    if (progressData.progress >= 100) {
                        console.log('✅ Task completed, stopping tracking');
                        isCompleted = true;
                        clearInterval(progressInterval);
                    }
                }
            } else {
                console.warn('⚠️ Progress fetch failed:', response.status);
            }
        } catch (error) {
            console.error('💥 Progress fetch error:', error);
        }
    }, 100); // Reduced to 800ms for better responsiveness

    console.log('✅ Progress tracking started');
    return progressInterval;
}

function updateProgressUI(progressBar, progressText, progressPercent, progressStatus, progressData) {
    console.log('🔄 UI Update called with:', progressData); // Debug log

    // ✅ Update progress bar
    if (progressBar && progressData.progress !== undefined) {
        progressBar.value = progressData.progress;
        progressBar.style.width = progressData.progress + '%';
        console.log('📊 Progress bar updated to:', progressData.progress + '%');
    }

    // ✅ Update progress percentage text
    if (progressPercent && progressData.progress !== undefined) {
        progressPercent.textContent = `${progressData.progress}%`;
    }

    // ✅ Update progress message
    if (progressText && progressData.message) {
        progressText.textContent = progressData.message;
    }

    // ✅ Update progress stage
    if (progressStatus && progressData.stage) {
        progressStatus.textContent = getProgressStage(progressData.stage);
    }

    // ✅ Force browser repaint
    if (progressBar) {
        progressBar.offsetHeight; // Trigger reflow
    }
}
function getProgressStage(stage) {
    const stages = {
        'initializing': 'Initializing...',
        'preparing_upload': 'Preparing Upload...',
        'uploading': 'Uploading to Cloud...',
        'downloading': 'Processing...',
        'compressing': 'Compressing PDF...',
        'ghostscript': 'Running Compression...',
        'alternative_compression': 'Optimizing...',
        'finalizing': 'Finalizing...',
        'completed': 'Completed!',
        'failed': 'Failed',
        'error': 'Error'
    };
    return stages[stage] || 'Processing...';
}



// Updated server compression estimation with progress

async function computeServerCompressionSizesTwoStep() {

    
    console.log('Computing server-side compression sizes using two-step method...');

    const form = document.getElementById('compressForm');
    const fileInput = document.getElementById('compress-file');
    const resultDiv = document.getElementById('result-compressForm');
    const compressionResults = document.getElementById('compression-results');
    const compressionSizes = document.getElementById('compression-sizes');


    const progressDiv = document.getElementById('progress-compressForm');
    const progressBar = document.getElementById('compressProgress');
    const progressText = document.getElementById('progress-text-compressForm');
    const progressPercent = document.getElementById('progress-percent-compressForm');
    const progressStatus = document.getElementById('progress-status-compressForm');
    const compbutton = document.getElementById('compress-submit-btn');
    const computeButton = document.getElementById('estimate-sizes-btn');


    if (!fileInput || !fileInput.files[0]) {
        const errorMsg = 'Please select a PDF file first.';
        if (resultDiv) {
            resultDiv.textContent = errorMsg;
            resultDiv.className = 'text-red-600';
        }
        return;
    }

    const file = fileInput.files[0];
    const originalSizeMB = (file.size / (1024 * 1024)).toFixed(2);
    console.log(`Checking size: ${originalSizeMB} MB`);
    
    if (originalSizeMB > 50) {
      resultDiv.textContent = 'Size exceeds 50MB';
      return
    } else {
      resultDiv.textContent = `File size is ${originalSizeMB} MB`;
      
    }
    
    


    // Disable button during computation
    if (computeButton) {
        computeButton.disabled = true;

        computeButton.innerHTML = '<i class="fas fa-calculator mr-2"></i> Computing...';
    }

    try {
        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="text-blue-600">🔄 Estimating server compression sizes...</div>
                <div class="text-sm text-gray-500 mt-1">Using First method</div>
            `;
        }

        // Initialize UI - SAME AS COMPRESSION
        progressDiv.style.display = 'block';
        updateProgressUI(progressBar, progressText, progressPercent, progressStatus, {
            progress: 0,
            message: 'Initializing uploading...',
            stage: 'initializing'
        });

        let taskId = null;
        let progressInterval = null;

        // STEP 1: Start estimation
        console.log('🚀 STEP 1: Starting estimation...');
        const formData = new FormData();
        formData.append('file', file);

        const startResponse = await fetch('/start_estimation', {
            method: 'POST',
            body: formData
        });

        if (!startResponse.ok) {
            const error = await startResponse.json();
            throw new Error(error.detail || 'Failed to start estimation');
        }

        const startData = await startResponse.json();
        taskId = startData.task_id;

        console.log('📨 Received task ID:', taskId);

        // STEP 2: Start real-time progress tracking
        console.log('🔄 STEP 2: Starting enhanced progress tracking');
        progressInterval = startProgressTracking(taskId, progressBar, progressText, progressPercent, progressStatus);

        // STEP 3: Wait for completion with timeout
        console.log('⏳ STEP 3: Waiting for completion...');
        await waitForEstimationCompletion(taskId, 180000); // 3 minute timeout

        // STEP 4: Get results
        console.log('📊 STEP 4: Getting results...');
        const resultResponse = await fetch(`/estimation_result/${taskId}`);

        if (!resultResponse.ok) {
            throw new Error('Failed to get estimation results');
        }

        const data = await resultResponse.json();

        // Display results
        displayCompressionResults(data, originalSizeMB, compressionSizes, compressionResults, resultDiv);

        // Final progress update
        updateProgressUI(progressBar, progressText, progressPercent, progressStatus, {
            progress: 100,
            message: 'Estimation completed successfully!',
            stage: 'completed'
        });

    } catch (error) {
        console.error('Estimation error:', error);

        // Stop progress tracking
        if (progressInterval) {
            clearInterval(progressInterval);
        }

        // Show error state
        updateProgressUI(progressBar, progressText, progressPercent, progressStatus, {
            progress: 0,
            message: `Error: ${error.message}`,
            stage: 'error'
        });

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Estimation failed: ${error.message}<br>
                <small>Falling back to client-side estimation...</small>
            </div>
        `;

        // Fallback to client-side estimation
        // await computeClientCompressionSizes();   

    } finally {
        // Re-enable button
        if (computeButton) {
            computeButton.disabled = false;
            computeButton.innerHTML = '<i class="fas fa-calculator mr-2"></i> Estimate Sizes';

        }
        compbutton.disabled = false;

        // Hide progress after delay
        setTimeout(() => {
            progressDiv.style.display = 'none';
        }, 3000);
    }
}

// Wait for estimation completion
async function waitForEstimationCompletion(taskId, timeoutMs = 180000) {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();
        const checkCompletion = async () => {
            try {
                if (Date.now() - startTime > timeoutMs) {
                    reject(new Error('Estimation timeout - process took too long'));
                    return;
                }

                const response = await fetch(`${BASE_URL}/progress/${taskId}`);
                if (response.ok) {
                    const progressData = await response.json();
                    if (progressData.progress >= 100) {
                        resolve(progressData);
                    } else {
                        // Continue polling
                        setTimeout(checkCompletion, 800);
                    }
                } else {
                    reject(new Error('Failed to check progress'));
                }
            } catch (error) {
                reject(error);
            }
        };

        checkCompletion();
    });
}


function displayCompressionResults(data, originalSizeMB, compressionSizes, compressionResults, resultDiv) {
    let sizesHTML = `
        <li class="font-semibold mb-2 text-gray-800">Original Size: ${originalSizeMB} MB</li>
        <li class="text-sm text-gray-600 mb-3">Method: First Method</li>
        <hr class="my-2 border-gray-300">
    `;

    // Presets matching the backend
    const presets = [
        { name: 'Screen Quality', key: 'screen', quality: '★★☆☆☆', desc: 'Maximum compression for web viewing' },
        { name: 'Ebook Quality', key: 'ebook', quality: '★★★☆☆', desc: 'Good compression for digital reading' },
        { name: 'Printer Quality', key: 'printer', quality: '★★★★☆', desc: 'Excellent quality, good compression' },
        { name: 'Prepress Quality', key: 'prepress', quality: '★★★★★', desc: 'Highest quality for professional printing' }
    ];

    // Display all presets that exist in the response
    presets.forEach(preset => {
        const estimate = data.estimates[preset.key];
        if (estimate !== undefined && estimate !== null && !isNaN(estimate)) {
            const sizeMB = parseFloat(estimate).toFixed(2);
            const originalSizeBytes = parseFloat(originalSizeMB) * 1024 * 1024;
            const estimatedSizeBytes = parseFloat(estimate) * 1024 * 1024;
            const savings = (((originalSizeBytes - estimatedSizeBytes) / originalSizeBytes) * 100).toFixed(1);
            const savingsColor = savings >= 50 ? 'text-green-600' :
                savings >= 20 ? 'text-yellow-600' :
                    savings >= 0 ? 'text-orange-600' : 'text-red-600';

            sizesHTML += `
                <li class="mb-2 p-3 bg-white rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors">
                    <div class="flex justify-between items-start">
                        <strong class="text-gray-800">${preset.name}</strong>
                        <span class="text-xs font-semibold text-blue-600">${preset.quality}</span>
                    </div>
                    <span class="text-sm text-gray-600">
                        📏 Size: <strong>${sizeMB} MB</strong><br>
                        💾 Reduction: <strong class="${savingsColor}">${savings}%</strong><br>
                        <small class="text-gray-500">${preset.desc}</small>
                    </span>
                </li>
            `;
        }
    });

    if (compressionSizes) {
        compressionSizes.innerHTML = sizesHTML;
    }

    if (compressionResults) {
        compressionResults.classList.remove('hidden');
    }

    // Show recommendation
    const recommendation = data.recommendation || 'ebook';
    const recommendationText = {
        'screen': 'Screen Quality',
        'ebook': 'Ebook Quality',
        'printer': 'Printer Quality',
        'prepress': 'Prepress Quality'
    };

    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Compression Estimates Ready!</strong><br>
                <small>Recommended: <strong>${recommendationText[recommendation]}</strong> based on actual Ghostscript compression</small><br>
                <small class="text-gray-500">These are real compression results, not estimates</small>
            </div>
        `;
    }
}

////////////////


async function computeClientCompressionSizes() {
    console.log('Computing accurate compression sizes with intelligent analysis...');

    const form = document.getElementById('compressForm');
    const fileInput = form.querySelector('input[type="file"]');
    const resultDiv = document.getElementById('result-compressForm');
    const compressionResults = document.getElementById('compression-results');
    const compressionSizes = document.getElementById('compression-sizes');
    const computeButton = document.getElementById('estimate-sizes-btn');
    const compbutton = document.getElementById('compress-submit-btn');

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

    const MAX_FILE_SIZE_MB = 250;
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        const errorMsg = `File size exceeds ${MAX_FILE_SIZE_MB}MB limit. Please choose a smaller file.`;
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
        // First analyze the PDF content
        const [pdfjs, pdfLib] = await pdfLibraryManager.loadLibraries(['pdfjs', 'pdfLib']);

        if (resultDiv) {
            resultDiv.innerHTML = `
              <div class="text-blue-600">🔄 Analyzing PDF content and computing sizes...</div>
            <progress id="compressProgress" value="0" max="100" class="w-1/2 h-2 mt-2 ml-8 mx-auto"></progress>
            <div id="compressionLogs" class="text-sm text-gray-600 mt-2 max-h-24 overflow-y-auto w-full"></div>
            `;
        }

        const progressBar = document.getElementById('compressProgress');
        const logDiv = document.getElementById('compressionLogs');

        const appendLog = (message) => {
            console.log(message);
            if (logDiv) {
                const logEntry = document.createElement('div');
                logEntry.textContent = message;
                logDiv.appendChild(logEntry);
                while (logDiv.children.length > 5) {
                    logDiv.removeChild(logDiv.firstChild);
                }
                logDiv.scrollTop = logDiv.scrollHeight;
            }
        };

        appendLog('Analyzing PDF content structure...');

        // Analyze PDF content
        const pdfAnalysis = await analyzePDFContent(file);
        appendLog(`Detected: ${pdfAnalysis.contentType}`);

        // Define presets with IMPROVED quality settings
        const presets = [
            {
                name: 'High Quality',
                strategy: 'quality',
                description: 'Best quality, minimal compression',
                settings: {
                    imageQuality: 0.95,
                    scale: 2.0,
                    useJPEG: false, // Use PNG for best quality
                    format: 'png'
                }
            },
            {
                name: 'Balanced',
                strategy: 'balanced',
                description: 'Good balance of quality and size',
                settings: {
                    imageQuality: 0.85,
                    scale: 1.5,
                    useJPEG: true,
                    format: 'jpeg'
                }
            },
            {
                name: 'Size Optimized',
                strategy: 'size',
                description: 'Better compression, good quality',
                settings: {
                    imageQuality: 0.75,
                    scale: 1.25,
                    useJPEG: true,
                    format: 'jpeg'
                }
            },
            {
                name: 'Smart Compression',
                strategy: 'smart',
                description: 'Intelligent content-based compression',
                settings: null // Uses analyzePDFContent results
            }
        ];

        let sizesHTML = `
            <li class="font-semibold mb-2 text-gray-800">Original Size: ${originalSizeMB} MB</li>
            <li class="text-sm text-gray-600 mb-3">Content Type: ${pdfAnalysis.contentType}</li>
            <hr class="my-2 border-gray-300">
        `;

        let computedCount = 0;

        for (const preset of presets) {
            appendLog(`Testing ${preset.name}...`);

            // Update progress
            const progress = Math.round((computedCount / presets.length) * 100);
            if (progressBar) progressBar.value = progress;
            if (compressionSizes) {
                compressionSizes.innerHTML = sizesHTML + `
                    <li class="text-blue-600">Computing... ${progress}% complete</li>
                    <li class="text-sm text-gray-500">Currently testing: ${preset.name}</li>
                `;
            }

            try {
                let compressedBlob;

                if (preset.strategy === 'smart') {
                    // Use intelligent analysis-based compression (same as main function)
                    if (pdfAnalysis.isTextHeavy && !pdfAnalysis.hasComplexImages) {
                        compressedBlob = await optimizeTextPDF(file, pdfAnalysis);
                    } else if (pdfAnalysis.isImageHeavy) {
                        compressedBlob = await compressImagePDF(file, pdfAnalysis);
                    } else {
                        compressedBlob = await compressMixedPDF(file, pdfAnalysis);
                    }
                } else {
                    // Use preset-based compression with improved settings
                    compressedBlob = await compressWithStrategyImproved(file, pdfAnalysis, preset.settings);
                }

                if (compressedBlob) {
                    const actualSizeMB = (compressedBlob.size / (1024 * 1024)).toFixed(2);
                    const savings = (((file.size - compressedBlob.size) / file.size) * 100).toFixed(1);

                    const savingsColor = savings >= 50 ? 'text-green-600' :
                        savings >= 20 ? 'text-yellow-600' :
                            savings >= 0 ? 'text-orange-600' : 'text-red-600';

                    const qualityIndicator = getQualityIndicator(preset.name);

                    sizesHTML += `
                        <li class="mb-2 p-3 bg-white rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors">
                            <div class="flex justify-between items-start">
                                <strong class="text-gray-800">${preset.name}</strong>
                                <span class="text-xs font-semibold ${getQualityColor(preset.name)}">${qualityIndicator}</span>
                            </div>
                            <span class="text-sm text-gray-600">
                                📏 Size: <strong>${actualSizeMB} MB</strong><br>
                                💾 Reduction: <strong class="${savingsColor}">${savings}%</strong><br>
                                📝 ${preset.description}
                            </span>
                        </li>
                    `;

                    appendLog(`✓ ${preset.name}: ${actualSizeMB}MB (${savings}% reduction)`);
                }
            } catch (error) {
                console.error(`Failed to compute ${preset.name}:`, error);
                sizesHTML += `
                    <li class="mb-2 p-3 bg-red-50 rounded-lg border border-red-200">
                        <strong class="text-red-700">${preset.name}</strong><br>
                        <span class="text-sm text-red-600">❌ Failed to compute: ${error.message}</span>
                    </li>
                `;
                appendLog(`✗ ${preset.name} failed: ${error.message}`);
            }

            computedCount++;
        }

        // Final progress update
        if (progressBar) progressBar.value = 100;
        if (compressionSizes) {
            compressionSizes.innerHTML = sizesHTML;
        }

        if (compressionResults) {
            compressionResults.classList.remove('hidden');
        }

        // Show appropriate message based on results
        if (resultDiv) {
            const hasPositiveSavings = sizesHTML.includes('Reduction:') &&
                !sizesHTML.includes('Reduction: -');

            let messageHTML;
            if (!hasPositiveSavings) {
                messageHTML = `
                    <div class="text-blue-600">
                        📊 <strong>PDF Analysis Complete</strong><br>
                        <small>This PDF is already optimized. Compression may not reduce size significantly.</small>
                    </div>
                `;
            } else {
                messageHTML = `
                    <div class="text-green-600">
                        ✅ <strong>Accurate Compression Estimation Completed!</strong><br>
                        <small>These estimates match actual download sizes. Choose your preferred balance.</small>
                    </div>
                `;
            }
            resultDiv.innerHTML = messageHTML;
        }

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
            computeButton.innerHTML = '<i class="fas fa-calculator mr-2"></i> Estimate Sizes';
            compbutton.disabled = false;
        }
        const compbutton = document.getElementById('compress-submit-btn');
        if (compbutton) compbutton.disabled = false;
    }
}

// IMPROVED: Unified compression function with better quality settings
async function compressWithStrategyImproved(file, pdfAnalysis, options) {
    const {
        imageQuality = 0.85,
        scale = 1.5,
        useJPEG = true,
        format = 'jpeg'
    } = options;

    const arrayBuffer = await file.arrayBuffer();
    const pdfjs = pdfLibraryManager.libraries.pdfjs.lib;
    const pdfLib = pdfLibraryManager.libraries.pdfLib.lib;
    const { PDFDocument } = pdfLib;

    const sourceDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
    const newPdfDoc = await PDFDocument.create();

    const totalPages = sourceDoc.numPages;

    for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
        const page = await sourceDoc.getPage(pageNum);
        const viewport = page.getViewport({ scale: 1.0 });

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d', {
            alpha: false,
            imageSmoothingEnabled: true,
            imageSmoothingQuality: 'high'
        });

        // Use improved scaling
        const actualScale = calculateOptimalScaleImproved(viewport, scale);
        canvas.width = Math.floor(viewport.width * actualScale);
        canvas.height = Math.floor(viewport.height * actualScale);

        // Better rendering settings
        context.fillStyle = 'white';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = 'high';

        const renderContext = {
            canvasContext: context,
            viewport: page.getViewport({ scale: actualScale }),
            enableWebGL: true
        };

        await page.render(renderContext).promise;

        // Use appropriate format with quality considerations
        let imageData;
        if (useJPEG) {
            imageData = canvas.toDataURL('image/jpeg', imageQuality);
        } else {
            imageData = canvas.toDataURL('image/png');
        }

        const base64Data = imageData.split(',')[1];
        const imageBytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));

        const image = useJPEG ?
            await newPdfDoc.embedJpg(imageBytes) :
            await newPdfDoc.embedPng(imageBytes);

        const newPage = newPdfDoc.addPage([viewport.width, viewport.height]);
        newPage.drawImage(image, {
            x: 0, y: 0,
            width: viewport.width,
            height: viewport.height,
        });

        canvas.remove();
    }

    const compressedBytes = await newPdfDoc.save({
        useObjectStreams: true,
        compressImages: true,
        imageQuality: useJPEG ? imageQuality : 1.0
    });

    return new Blob([compressedBytes], { type: 'application/pdf' });
}

// IMPROVED: Better scale calculation for quality
function calculateOptimalScaleImproved(viewport, baseScale) {
    const baseWidth = viewport.width;
    const baseHeight = viewport.height;

    // Higher base scales for better quality
    const sizeFactor = Math.max(baseWidth, baseHeight);

    if (sizeFactor > 2000) return baseScale * 0.9;    // Large documents
    if (sizeFactor > 1000) return baseScale * 1.0;    // Medium documents  
    if (sizeFactor > 500) return baseScale * 1.1;     // Small documents
    return baseScale * 1.2;                           // Very small: higher DPI
}

// Helper functions for UI
function getQualityIndicator(presetName) {
    const indicators = {
        'High Quality': '★★★★★',
        'Balanced': '★★★★☆',
        'Size Optimized': '★★★☆☆',
        'Smart Compression': '★★★★☆'
    };
    return indicators[presetName] || '★★★☆☆';
}

function getQualityColor(presetName) {
    const colors = {
        'High Quality': 'text-green-600',
        'Balanced': 'text-blue-600',
        'Size Optimized': 'text-orange-600',
        'Smart Compression': 'text-purple-600'
    };
    return colors[presetName] || 'text-gray-600';
}

// Update the main compress function to use the same improved logic
async function compressPDFClientSide() {
    console.log('Starting intelligent PDF compression...');

    const [pdfjs, pdfLib] = await pdfLibraryManager.loadLibraries(['pdfjs', 'pdfLib']);

    const form = document.getElementById('compressForm');
    const fileInput = form.querySelector('input[type="file"]');
    const resultDiv = document.getElementById('result-compressForm');
    const progressDiv = document.getElementById('progress-compressForm');
    const progressText = document.getElementById('progress-text-compressForm');
    const submitButton = form.querySelector('button[type="button"]');

    // Get selected preset
    const presetSelect = document.getElementById('compress-preset');
    const selectedPreset = presetSelect ? presetSelect.value : 'Balanced';
    const computeButton = document.getElementById('estimate-sizes-btn');

    if (!fileInput?.files?.length) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];
    const originalSizeMB = (file.size / (1024 * 1024)).toFixed(2);

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Analyzing PDF content...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Analyzing...';

    try {
        // Analyze PDF content
        progressText.textContent = 'Analyzing PDF structure...';
        const pdfAnalysis = await analyzePDFContent(file);

        console.log('PDF Analysis:', pdfAnalysis);

        // Map preset names to improved strategies
        const presetStrategies = {
            'High Quality': {
                strategy: 'quality',
                imageQuality: 0.95,
                scale: 2.0,
                useJPEG: false
            },
            'Balanced': {
                strategy: 'balanced',
                imageQuality: 0.85,
                scale: 1.5,
                useJPEG: true
            },
            'Size Optimized': {
                strategy: 'size',
                imageQuality: 0.75,
                scale: 1.25,
                useJPEG: true
            },
            'Smart Compression': {
                strategy: 'smart'
            }
        };

        let compressedBlob;
        const strategy = presetStrategies[selectedPreset];

        if (strategy.strategy === 'smart') {
            // Use intelligent analysis-based compression
            if (pdfAnalysis.isTextHeavy && !pdfAnalysis.hasComplexImages) {
                compressedBlob = await optimizeTextPDF(file, pdfAnalysis);
            } else if (pdfAnalysis.isImageHeavy) {
                compressedBlob = await compressImagePDF(file, pdfAnalysis);
            } else {
                compressedBlob = await compressMixedPDF(file, pdfAnalysis);
            }
        } else {
            // Use preset-based compression with improved settings
            compressedBlob = await compressWithStrategyImproved(file, pdfAnalysis, strategy);
        }

        if (!compressedBlob) {
            throw new Error('Compression failed');
        }

        // Calculate results
        const compressedSizeMB = (compressedBlob.size / (1024 * 1024)).toFixed(2);
        const savings = (((file.size - compressedBlob.size) / file.size) * 100).toFixed(1);

        // Download
        const filename = `compressed_${file.name.replace('.pdf', '')}_${selectedPreset.replace(' ', '_')}.pdf`;
        downloadBlob(compressedBlob, filename);

        // Show results
        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Compression Successful!</strong><br>
                📁 Original: ${originalSizeMB}MB → Compressed: ${compressedSizeMB}MB<br>
                💾 Size reduction: <strong class="${savings >= 0 ? 'text-green-600' : 'text-red-600'}">${savings}%</strong><br>
                ⚙️ Preset: <strong>${selectedPreset}</strong><br>
                📊 Content: ${pdfAnalysis.contentType}<br>
                <small class="text-gray-500">Quality: ${getQualityIndicator(selectedPreset)}</small>
            </div>
        `;

    } catch (error) {
        console.error('Compression failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Compression failed: ${error.message}
            </div>
        `;
    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Compress PDF';
        computeButton.disabled = false;
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////

async function updateFileSize() {
    const fileInput = document.getElementById('compress-file');
    const fileNameDisplay = document.getElementById('compress-file-name');
    const fileSizeDisplay = document.getElementById('original-file-size');
    const fileInfo = document.getElementById('file-info');

    if (fileInput && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);

        fileNameDisplay.textContent = file.name;
        fileSizeDisplay.textContent = `Original File Size: ${sizeMB} MB`;

        const compressionType = document.querySelector('input[name="compression_type"]:checked');
        if (compressionType && compressionType.value === 'server') {
            fileInfo.innerHTML = `<i class="fas fa-server mr-1 text-blue-500"></i>Compression`;
        } else {
            fileInfo.innerHTML = `<i class="fas fa-desktop mr-1 text-purple-500"></i>Compression (Fully private)`;
        }

        // Auto-compute sizes for files under 50MB
        // if (file.size < 50 * 1024 * 1024) {
        //     fileInfo.innerHTML = `<i class="fas fa-sync-alt mr-1 animate-spin"></i> Analyzing file...`;
        //     setTimeout(() => {
        //         computeAllCompressionSizes();
        //     }, 1000);
        // }
    } else {
        fileNameDisplay.textContent = 'No file selected';
        fileSizeDisplay.textContent = 'Original File Size: Not selected';
        fileInfo.innerHTML = '';
    }
}



// // /// //  // NEW ADD SIGNATURE FOR CLIENT SIDE

// async function loadpdflibray() {
//     const [pdfjs, pdfLib] = await pdfLibraryManager.loadLibraries([
//         'pdfjs', 'pdfLib'
//     ]);
//     console.log("library loaded on file upload");

// }


async function processSignatureClientSide() {
    console.log('Starting client-side signature processing...');
    // const [pdfLib, pdfjs, fileSaver] = await pdfLibraryManager.loadLibraries([
    //     'pdfLib', 'pdfjs', 'fileSaver'
    // ]);


    const form = document.getElementById('signatureForm');
    const pdfFileInput = document.getElementById('signature-pdf-file');
    const signatureFileInput = document.getElementById('signature-image-file');
    const selectedPagesInput = document.getElementById('signature-selected-pages');
    const sizeSelect = document.getElementById('signature-size');
    const positionSelect = document.getElementById('signature-position');
    const alignmentSelect = document.getElementById('signature-alignment');
    const removeBgCheckbox = document.getElementById('remove-bg');
    const addsignbutton = document.getElementById('addsign');

    // Safely get progress elements with null checks
    const progressDiv = document.getElementById('progress-signatureForm');
    const progressText = document.getElementById('progress-text-signatureForm');
    const resultDiv = document.getElementById('result-signatureForm');
    const submitButton = form ? form.querySelector('button') : null;


    const pdfFile = pdfFileInput.files[0];
    const selectedPages = selectedPagesInput.value.split(',').map(p => parseInt(p.trim()));
    const size = sizeSelect ? sizeSelect.value : 'medium';
    const position = positionSelect ? positionSelect.value : 'bottom';
    const alignment = alignmentSelect ? alignmentSelect.value : 'center';
    const removeBg = removeBgCheckbox ? removeBgCheckbox.checked : false;

    // Validation with better error handling
    // Check if all required elements exist first
    if (!pdfFileInput || !signatureFileInput || !selectedPagesInput || !sizeSelect || !positionSelect || !alignmentSelect) {
        if (resultDiv) {
            resultDiv.textContent = 'Form elements not loaded properly. Please refresh the page.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }

    // Then check if files are selected
    if (!pdfFileInput.files[0]) {
        if (resultDiv) {
            resultDiv.textContent = 'Please select a PDF file.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }

    if (!signatureFileInput.files[0]) {
        if (resultDiv) {
            resultDiv.textContent = 'Please select a signature image.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }

    if (!selectedPagesInput.value) {
        if (resultDiv) {
            resultDiv.textContent = 'Please select at least one page.';
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

    const pdfSizeMB = pdfFile.size / (1024 * 1024);
    if (pdfSizeMB > 200) { // Fixed: 200MB limit
        if (resultDiv) {
            resultDiv.textContent = 'PDF file exceeds 200MB limit.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }

    const sigSizeMB = signatureFile.size / (1024 * 1024);
    if (sigSizeMB > 20) {
        if (resultDiv) {
            resultDiv.textContent = 'Signature image exceeds 20MB limit.';
            resultDiv.classList.add('text-red-600');
        }
        return;
    }



    // if (submitButton) submitButton.disabled = true;
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = 'Processing...'; // for <button>
        // submitButton.value = 'Processing...'; // use this if it's an <input type="submit">
    }

    resultDiv.textContent = ""
    const [pdfjs, pdfLib] = await pdfLibraryManager.loadLibraries([
        'pdfjs', 'pdfLib'
    ]);




    try {

        // addsignbutton.disabled = true;
        // addsignbutton.textContent = 'Processing...';
        // Show progress safely

        if (progressDiv) progressDiv.style.display = 'block';
        if (progressText) progressText.textContent = 'Processing signature...';


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
            resultDiv.textContent = 'Signature added successfully!';
            resultDiv.classList.remove('text-red-600');
            resultDiv.classList.add('text-green-600');
        }

    } catch (error) {
        console.error('signature error:', error);

        if (resultDiv) {
            resultDiv.textContent = `Error: ${error.message}. Trying server-side fallback...`;
            resultDiv.classList.add('text-red-600');
        }

        // Fallback to server-side processing 
        // await fallbackToServerSideSignature(form);

    } finally {
        // Clean up safely
        // if (submitButton) submitButton.disabled = false;
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = `<i class="fas fa-signature mr-2"></i> Add Signature`;
        }
        if (progressDiv) progressDiv.style.display = 'none';
        if (progressText) progressText.textContent = '';
        // addsignbutton.disabled = false;
        // addsignbutton.textContent = 'Add Signature';
    }
}



async function addSignatureClientSide(pdfFile, signatureFile, pages, size, position, alignment, removeBg = false) {
    // Check if PDF-LIB is available
    if (!pdfLibraryManager.libraries.pdfLib.loaded) {
        throw new Error('PDF library not loaded. Please refresh the page.');
    }

    const pdfLib = pdfLibraryManager.libraries.pdfLib.lib;

    if (!pdfLib) {
        throw new Error('PDF library not loaded. Please ensure libraries are loaded first.');
    }

    const { PDFDocument } = pdfLib;

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
            xsmall: 0.08,
            small: 0.15,
            medium: 0.25,
            large: 0.35,
            xlarge: 0.5
        };
        const scale = sizeFactors[size] || 0.25;
        console.log(" check scale size", scale);
        // print("check scale size",scale)


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
            y = 25;
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

        img.onload = function () {
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

                    // Remove white/light background 
                    if (r > 200 && g > 200 && b > 200) {
                        data[i + 3] = 0; // 
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

        img.onerror = function () {
            reject(new Error('Failed to load signature image'));
        };

        img.src = URL.createObjectURL(imageFile);
    });
}



// 
//////////////////////// merge pdf opertion client side

function validateFilesForClientMerge(files) {
    const maxFiles = 200;
    const maxTotalSizeMB = 500;
    const maxFileSizeMB = 200;

    if (files.length > maxFiles) {
        return {
            valid: false,
            message: `Too many files. Merge supports maximum ${maxFiles} files.`
        };
    }

    let totalSize = 0;
    for (let file of files) {
        const fileSizeMB = file.size / (1024 * 1024);
        if (fileSizeMB > maxFileSizeMB) {
            return {
                valid: false,
                message: `File "${file.name}" is too large (${fileSizeMB.toFixed(2)}MB). Maximum file size is ${maxFileSizeMB}MB.`
            };
        }
        totalSize += file.size;
    }

    const totalSizeMB = totalSize / (1024 * 1024);
    if (totalSizeMB > maxTotalSizeMB) {
        return {
            valid: false,
            message: `Total file size (${totalSizeMB.toFixed(2)}MB) exceeds maximum ${maxTotalSizeMB}MB limit.`
        };
    }

    return { valid: true, totalSizeMB: totalSizeMB };
}


async function mergePDFsClientSide() {
    console.log('Starting client-side PDF merge...');

    const form = document.getElementById('mergeForm');
    const fileInput = document.getElementById('merge-files');
    const resultDiv = document.getElementById('result-mergeForm');
    const progressDiv = document.getElementById('progress-mergeForm');
    const progressText = document.getElementById('progress-text-mergeForm');
    const submitButton = document.getElementById('merge-submit-btn');

    // Validation
    if (!fileInput || !fileInput.files || fileInput.files.length < 2) {
        alert('Please select at least 2 PDF files.');
        return;
    }


    const orderedFiles = getFilesInDOMOrder();

    if (orderedFiles.length < 2) {
        alert('Please select at least 2 PDF files.');
        return;
    }

    const validation = validateFilesForClientMerge(orderedFiles);
    if (!validation.valid) {
        alert(validation.message);
        return;
    }

    console.log("🎯 Files to merge in order:");
    orderedFiles.forEach((file, index) => {
        console.log(`${index + 1}. ${file.name}`);
    });

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting merge...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-object-group mr-2"></i> Merging...';

    try {
        const [pdfLib] = await pdfLibraryManager.loadLibraries([
            'pdfLib'
        ]);

        // Process files in DOM order
        const pdfDocs = [];
        for (let i = 0; i < orderedFiles.length; i++) {
            const progress = Math.round((i / orderedFiles.length) * 50);
            progressText.textContent = `Loading PDF files... (${progress}%)`;

            const file = orderedFiles[i];
            console.log(`Loading PDF ${i + 1}/${orderedFiles.length}:`, file.name);

            const arrayBuffer = await file.arrayBuffer();
            const pdfDoc = await pdfLib.PDFDocument.load(arrayBuffer);
            pdfDocs.push(pdfDoc);
        }

        progressText.textContent = 'Merging PDFs... (50%)';

        // Create new PDF document
        const mergedPdf = await pdfLib.PDFDocument.create();

        // Copy pages from all PDFs in DOM order
        for (let i = 0; i < pdfDocs.length; i++) {
            const progress = 50 + Math.round((i / pdfDocs.length) * 45);
            progressText.textContent = `Merging PDFs... (${progress}%)`;

            const pdfDoc = pdfDocs[i];
            const pages = await mergedPdf.copyPages(pdfDoc, pdfDoc.getPageIndices());
            pages.forEach(page => mergedPdf.addPage(page));

            console.log(`Added file: ${orderedFiles[i].name}`);
        }

        progressText.textContent = 'Finalizing merge... (95%)';

        // Save the merged PDF
        const mergedPdfBytes = await mergedPdf.save();
        const mergedBlob = new Blob([mergedPdfBytes], { type: 'application/pdf' });

        progressText.textContent = 'Downloading... (100%)';

        // Download the merged PDF
        const filename = `merged_${Date.now()}.pdf`;
        const url = URL.createObjectURL(mergedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Show success message
        let totalSize = 0;
        for (let file of orderedFiles) {
            totalSize += file.size;
        }
        const totalSizeMB = totalSize / (1024 * 1024);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDFs Merged Successfully!</strong><br>
                📁 Merged ${orderedFiles.length} files in your specified order (${totalSizeMB.toFixed(2)}MB total)<br>
                📋 Order maintained: ${orderedFiles.map(f => f.name).join(' → ')}
            </div>
        `;

        console.log(' merge completed successfully.');

    } catch (error) {
        console.error(' merge failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Merge failed: ${error.message}
            </div>
        `;
    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-object-group mr-2"></i> Merge PDFs';
    }
}

// // Get files in exact DOM order
function getFilesInDOMOrder() {
    const fileInput = document.getElementById('merge-files');
    const fileItems = document.querySelectorAll('#file-list .file-item');

    if (!fileInput || !fileInput.files || fileItems.length === 0) {
        return [];
    }

    // Create a map of filename to file object for quick lookup
    const filesMap = {};
    Array.from(fileInput.files).forEach(file => {
        filesMap[file.name] = file;
    });

    // Get files in exact DOM order by matching filenames
    const orderedFiles = Array.from(fileItems).map(fileItem => {
        const fileName = fileItem.querySelector('span').textContent.trim();
        return filesMap[fileName];
    }).filter(file => file !== undefined);

    console.log("📁 DOM Order:", Array.from(fileItems).map(item => item.querySelector('span').textContent.trim()));
    console.log("📄 Final ordered files:", orderedFiles.map(f => f.name));

    return orderedFiles;
}

//some CSS for better styling of the file items
const additionalStyles = `
    .file-item {
        transition: all 0.3s ease;
    }
    
    .file-item:hover {
        background-color: #f9fafb;
        border-color: #3b82f6;
    }
    
    .remove-file {
        min-width: 32px;
    }
    
    .file-item span {
        word-break: break-all;
    }
`;

// Add the styles to the document
if (!document.querySelector('#merge-file-styles')) {
    const styleEl = document.createElement('style');
    styleEl.id = 'merge-file-styles';
    styleEl.textContent = additionalStyles;
    document.head.appendChild(styleEl);
}




// ////////////////////

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
    const presetSelect = document.getElementById('compress-preset');
    const customOptions = document.getElementById('custom-compress-options');

    if (presetSelect.value === 'Custom') {
        customOptions.classList.remove('hidden');
    } else {
        customOptions.classList.add('hidden');
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



// /////////////////////////////////
function setupCompressionType() {
    const serverRadio = document.getElementById('server-compression');
    const clientRadio = document.getElementById('client-compression');
    const serverOptions = document.getElementById('server-options');
    const clientOptions = document.getElementById('client-options');
    const serverlabel = document.getElementById('serverlabel');
    const clientlabel = document.getElementById('clientlabel');
    const estimatesizebuttonforhideing = document.getElementById('estimate-sizes-btn');
    const serverlabelforcolorchange = document.querySelector('label[for="server-compression"]');
    const clientlabelforcolorchange = document.querySelector('label[for="client-compression"]');

    function updateOptions() {
        if (serverRadio.checked) {
            estimatesizebuttonforhideing.style.display= 'none'  // this is hiding estimate size in server option
            serverOptions.classList.remove('hidden');
            clientOptions.classList.add('hidden');
            serverlabel.style.backgroundColor = '#d1e7dd'; // light green
            serverlabelforcolorchange.style.backgroundColor = '#d1e7dd';
            clientlabelforcolorchange.style.backgroundColor = '';

            clientlabel.style.backgroundColor = ''; // reset
        } else {
            estimatesizebuttonforhideing.style.display= 'flex'
            serverOptions.classList.add('hidden');
            clientOptions.classList.remove('hidden');
            clientlabel.style.backgroundColor = '#cff4fc'; // light blue
            serverlabel.style.backgroundColor = ''; // reset
            serverlabelforcolorchange.style.backgroundColor = ''; // reset
            clientlabelforcolorchange.style.backgroundColor = '#cff4fc'
        }
    }

    if (serverRadio && clientRadio) {
        serverRadio.addEventListener('change', updateOptions);
        clientRadio.addEventListener('change', updateOptions);
        updateOptions(); // Initial call
    }
}
////////////////////////////////////////////

async function processServerCompressionTwoStep(form, resultDiv, progressDiv, progressBar, progressText, progressPercent, progressStatus, submitButton) {
    const fileInput = form.querySelector('#compress-file');
    const preset = document.getElementById('server-preset')?.value || 'ebook';
    const computeButton = document.getElementById('estimate-sizes-btn');

    if (!fileInput || !fileInput.files[0]) {
        resultDiv.textContent = 'Please select a PDF file.';
        resultDiv.classList.add('text-red-600');
        return;
    }

    const file = fileInput.files[0];

    // Initialize UI
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Starting...';
    progressDiv.style.display = 'block';
    updateProgressUI(progressBar, progressText, progressPercent, progressStatus, {
        progress: 0,
        message: 'Initializing uploading...',
        stage: 'initializing'
    });

    let taskId = null;
    let progressInterval = null;

    try {
        // STEP 1: Start compression
        console.log('🚀 STEP 1: Starting compression...');
        const formData = new FormData();
        formData.append('file', file);
        formData.append('preset', preset);

        const startResponse = await fetch('/start_compression', {
            method: 'POST',
            body: formData
        });

        if (!startResponse.ok) {
            const error = await startResponse.json();
            throw new Error(error.detail || 'Failed to start compression');
        }

        const startData = await startResponse.json();
        taskId = startData.task_id;

        console.log('📨 Received task ID:', taskId);

        // STEP 2: Start real-time progress tracking
        console.log('🔄 STEP 2: Starting enhanced progress tracking');
        progressInterval = startProgressTracking(taskId, progressBar, progressText, progressPercent, progressStatus);

        // STEP 3: Wait for completion with timeout
        console.log('⏳ STEP 3: Waiting for completion...');
        await waitForCompressionCompletion(taskId, 300000); // 5 minute timeout

        // STEP 4: Download result
        console.log('📥 STEP 4: Downloading result...');
        await downloadCompressedResult(taskId, resultDiv);

    } catch (error) {
        console.error('Compression error:', error);

        // Stop progress tracking
        if (progressInterval) {
            clearInterval(progressInterval);
        }

        // Show error state
        updateProgressUI(progressBar, progressText, progressPercent, progressStatus, {
            progress: 0,
            message: `Error: ${error.message}`,
            stage: 'error'
        });

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Compression failed: ${error.message}
            </div>
        `;
    } finally {
        // Re-enable button
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-compress-alt mr-2"></i> Compress PDF';
        computeButton.disabled = false;

    }
}

// Enhanced completion waiter with timeout
async function waitForCompressionCompletion(taskId, timeoutMs = 300000) {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();
        const checkCompletion = async () => {
            try {
                if (Date.now() - startTime > timeoutMs) {
                    reject(new Error('Compression timeout - process took too long'));
                    return;
                }

                const response = await fetch(`${BASE_URL}/progress/${taskId}`);
                if (response.ok) {
                    const progressData = await response.json();
                    if (progressData.progress >= 100) {
                        resolve(progressData);
                    } else {
                        // Continue polling
                        setTimeout(checkCompletion, 1000);
                    }
                } else {
                    reject(new Error('Failed to check progress'));
                }
            } catch (error) {
                reject(error);
            }
        };

        checkCompletion();
    });
}
// NEW: Download compressed result
async function downloadCompressedResult(taskId, resultDiv) {
    const response = await fetch(`${BASE_URL}/download_compressed/${taskId}`);

    if (!response.ok) {
        throw new Error('Failed to download compressed file');
    }

    // Get filename from headers
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = 'compressed.pdf';
    if (contentDisposition) {
        const match = contentDisposition.match(/filename="(.+)"|filename=([^;]+)/i);
        if (match) filename = match[1] || match[2];
    }

    const blob = await response.blob();

    // Trigger download
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    // Show success message
    const originalSize = response.headers.get('X-Original-Size');
    const compressedSize = response.headers.get('X-Compressed-Size');
    const savings = response.headers.get('X-Savings-Percent');

    if (originalSize && compressedSize && savings) {
        const originalSizeMB = (parseInt(originalSize) / (1024 * 1024)).toFixed(2);
        const compressedSizeMB = (parseInt(compressedSize) / (1024 * 1024)).toFixed(2);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Compression Successful!</strong><br>
                📁 Original: ${originalSizeMB}MB → Compressed: ${compressedSizeMB}MB<br>
                💾 Size reduction: <strong class="text-green-600">${savings}%</strong><br>
                ⚙️ Method: <strong>First</strong>
            </div>
        `;
    }
}

/////////////////////////////////
async function processPDF(endpoint, formId) {
    console.log(`Processing PDF for endpoint: ${endpoint}, form: ${formId}`);
    const form = document.getElementById(formId);
    const resultDiv = document.getElementById(`result-${formId}`);
    const submitButton = form.querySelector('button');

    const progressDiv = document.getElementById('progress-compressForm');
    const progressBar = document.getElementById('compressProgress');
    const progressText = document.getElementById('progress-text-compressForm');
    const progressPercent = document.getElementById('progress-percent-compressForm');
    const progressStatus = document.getElementById('progress-status-compressForm');
    const computeButton = document.getElementById('estimate-sizes-btn');
    const pdftoexcelbutton = document.getElementById('pdftoexcelbutton');
    const pdftowordbutton = document.getElementById('pdftowordbutton');
    

    if (!form) {
        console.error(`Form with ID ${formId} not found`);
        resultDiv.textContent = 'Form not found.';
        resultDiv.classList.add('text-red-600');
        return;
    }

    if (!(await validateForm(form, endpoint, resultDiv))) {
        console.log('Validation failed');
        return;
    }

    // Handle compression type selection
    if (endpoint === 'compress_pdf') {
        const compressionType = document.querySelector('input[name="compression_type"]:checked');
        if (compressionType && compressionType.value === 'client') {
            await compressPDFClientSide();
            return;
        }

        // NEW: Use two-step process for server compression
        await processServerCompressionTwoStep(form, resultDiv, progressDiv, progressBar, progressText, progressPercent, progressStatus, submitButton);
        return;
    }

    let processingButton = null;
    let originalHTML = '';

    if (endpoint === 'convert_pdf_to_excel' && pdftoexcelbutton) {
        processingButton = pdftoexcelbutton;
        originalHTML = pdftoexcelbutton.innerHTML;
        pdftoexcelbutton.disabled = true;
        pdftoexcelbutton.innerHTML = '<i class="fas fa-sync-alt fa-spin mr-2"></i> Processing...';
    }
    else if (endpoint === 'convert_pdf_to_word' && pdftowordbutton) {
        processingButton = pdftowordbutton;
        originalHTML = pdftowordbutton.innerHTML;
        pdftowordbutton.disabled = true;
        pdftowordbutton.innerHTML = '<i class="fas fa-sync-alt fa-spin mr-2"></i> Processing...';
    }
    else {
        processingButton = submitButton;
        originalHTML = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-sync-alt fa-spin mr-2"></i> Processing...';
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
        formData.append('font_color', fontColor);
        formData.append('font_family', fontFamily);
        formData.append('font_weight', fontWeight);
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
    } else if (endpoint === 'compress_pdf_server') {
        const fileInput = form.querySelector('#compress-file');
        const preset = document.getElementById('server-preset')?.value || 'ebook';

        if (!fileInput || !fileInput.files[0]) {
            resultDiv.textContent = 'Please select a PDF file.';
            resultDiv.classList.add('text-red-600');
            return;
        }

        formData.append('file', fileInput.files[0]);
        formData.append('preset', preset);

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
    }



    else {
        const conversionTypeInput = form.querySelector('input[name="conversionType"]:checked');
        formData = new FormData(form);
        if (conversionTypeInput) {
            const conversionType = conversionTypeInput.value;
            formData.append('conversion_type', conversionType);
            console.log('Conversion type:', conversionType);
        }
    }



    console.log('Sending request to:', `${BASE_URL}/${endpoint}`);
    console.log('FormData contents:');
    for (const [key, value] of formData.entries()) {
        console.log(`${key}: ${value instanceof File ? value.name : value}`);
    }

    submitButton.disabled = true;

    // Initialize progress tracking
    if (progressDiv) progressDiv.style.display = 'block';
    if (progressBar) progressBar.value = 0;
    if (progressText) progressText.textContent = 'Starting...';
    if (progressPercent) progressPercent.textContent = '0%';
    if (progressStatus) progressStatus.textContent = 'Initializing';

    let progressInterval = null;
    let taskId = null;

    try {

        console.log('🚀 Sending request AND starting progress simulation immediately');

        // Start simulated progress immediately (will be overridden by real progress)
        progressInterval = simulateProgress(progressBar, progressText, progressPercent, progressStatus);

        const response = await fetch(`${BASE_URL}/${endpoint}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Server error');
        }

        // Get task ID from response headers for progress tracking
        taskId = response.headers.get('X-Task-ID');

        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
        const fileInput = document.getElementById('compress-file');
        const file = fileInput?.files[0];
        console.log('📨 Received task ID:', taskId);
        console.log('🔄 TESTING FOR ONLY real progress tracking for large file');
        progressInterval = startProgressTracking(taskId, progressBar, progressText, progressPercent, progressStatus);

        // Determine if we should use real progress or simulated
        // const shouldUseRealProgress = taskId && file && file.size > 10 * 1024 * 1024;

        // if (shouldUseRealProgress) {
        //     console.log('🔄 Switching to real progress tracking for large file');
        //     progressInterval = startProgressTracking(taskId, progressBar, progressText, progressPercent, progressStatus);
        // } else {
        //     console.log('⚡ Using simulated progress for fast compression');
        //     progressInterval = simulateProgress(progressBar, progressText, progressPercent, progressStatus);
        // }


        // Handle the file download WITHOUT blocking on blob()
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'output.pdf';
        if (contentDisposition) {
            const match = contentDisposition.match(/filename="(.+)"|filename=([^;]+)/i);
            if (match) filename = match[1] || match[2];
        }

        const blob = await response.blob();

        // Update UI to show completion
        if (progressBar) progressBar.value = 100;
        if (progressPercent) progressPercent.textContent = '100%';
        if (progressText) progressText.textContent = 'Download complete!';
        if (progressStatus) progressStatus.textContent = 'Completed';


        // Get compression results from headers
        if (endpoint === 'compress_pdf_server') {
            const originalSize = response.headers.get('X-Original-Size');
            const compressedSize = response.headers.get('X-Compressed-Size');
            const savings = response.headers.get('X-Savings-Percent');

            if (originalSize && compressedSize && savings) {
                const originalSizeMB = (parseInt(originalSize) / (1024 * 1024)).toFixed(2);
                const compressedSizeMB = (parseInt(compressedSize) / (1024 * 1024)).toFixed(2);

                resultDiv.innerHTML = `
                    <div class="text-green-600">
                        ✅ <strong>Server Compression Successful!</strong><br>
                        📁 Original: ${originalSizeMB}MB → Compressed: ${compressedSizeMB}MB<br>
                        💾 Size reduction: <strong class="text-green-600">${savings}%</strong><br>
                        ⚙️ Method: <strong>Ghostscript</strong>
                    </div>
                `;
            } else {
                resultDiv.textContent = 'PDF compressed successfully with Ghostscript!';
                resultDiv.classList.add('text-green-600');
            }
        } else {
            resultDiv.textContent = 'Processing completed successfully!';
            resultDiv.classList.add('text-green-600');
        }

        // Trigger download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Error:', error);

        // Stop progress tracking on error
        if (progressInterval) {
            clearInterval(progressInterval);
        }

        // Show error in progress
        if (progressText) progressText.textContent = `Error: ${error.message}`;
        if (progressBar) progressBar.value = 0;
        if (progressStatus) progressStatus.textContent = 'Failed';

        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.classList.add('text-red-600');
    } finally {
        submitButton.disabled = false;

        if (processingButton) {
            processingButton.disabled = false;
            processingButton.innerHTML = originalHTML;
        }

        // Clean up progress tracking after delay
        setTimeout(() => {
            if (progressDiv) progressDiv.style.display = 'none';
            if (progressBar) progressBar.value = 0;
            if (progressText) progressText.textContent = '';
            if (progressStatus) progressStatus.textContent = '';
        }, 3000);
    }
}




// NEW: Fallback progress simulation
function simulateProgress(progressBar, progressText, progressPercent, progressStatus) {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 10 + 5; // Variable progress
        if (progress > 85) progress = 85; // Cap at 85% until real progress takes over

        if (progressBar) progressBar.value = progress;
        if (progressPercent) progressPercent.textContent = `${Math.round(progress)}%`;
        if (progressText) progressText.textContent = getProgressMessage(progress);
        if (progressStatus) progressStatus.textContent = getProgressStatus(progress);
    }, 500);

    return interval;
}

// NEW: Helper functions
function getProgressStatus(progress) {
    if (progress < 20) return 'Uploading';
    if (progress < 40) return 'Processing';
    if (progress < 70) return 'Compressing';
    if (progress < 90) return 'Finalizing';
    return 'Completed';
}
function getProgressMessage(progress) {
    const messages = {
        0: 'Starting compression process...',
        10: 'Uploading file to server...',
        30: 'Analyzing PDF content...',
        50: 'Compressing with Ghostscript...',
        70: 'Optimizing compression...',
        90: 'Finalizing and preparing download...',
        100: 'Compression complete!'
    };

    return messages[progress] || 'Processing...';
}

function stopProgressTracking() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    currentTaskId = null;
}

// NEW: Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopProgressTracking();
});


function updateFileOrder(fileElements) {
    const fileOrder = Array.from(fileElements).map(fileElement => {
        return fileElement.dataset.fileIndex;
    });

    const fileOrderInput = document.getElementById('merge-file-order');
    if (fileOrderInput) {
        fileOrderInput.value = fileOrder.join(',');
        console.log("Updated file order:", fileOrderInput.value);
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

    //  CHECKS FOR PDF WORD EXCEL

    if (endpoint === 'convert_pdf_to_word' || endpoint === 'convert_pdf_to_excel') {
        const file = files[0];
    
        // Check file size
        if (file.size > MAX_FILE_SIZE_PDFWORDEXCEL) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            resultDiv.textContent = `File ${file.name} (${sizeMB}MB) exceeds 10MB limit for conversion.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
    
        // Check file type
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
    
        // Check page count - FIXED CODE
        try {
            const arrayBuffer = await file.arrayBuffer();
            const [pdfjs] = await pdfLibraryManager.loadLibraries(['pdfjs']);
            
            // ✅ CORRECT: Load the PDF document first
            const pdfDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
            const totalPages = pdfDoc.numPages; // ✅ Get pages from the loaded document
    
            if (totalPages > PDFWORDEXCEL_MAX_PAGES) {
                resultDiv.textContent = `File ${file.name} has ${totalPages} pages. Maximum ${PDFWORDEXCEL_MAX_PAGES} pages allowed for conversion.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
        } catch (err) {
            console.error('Error counting pages:', err);
            resultDiv.textContent = 'Error loading PDF file. Please ensure it\'s a valid PDF.';
            resultDiv.classList.add('text-red-600');
            return false;
        }
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

        // In the validateForm function, update the compress_pdf section:

    } else if (endpoint === 'compress_pdf') {
        const file = files[0];
        const sizeMB = file.size / (1024 * 1024);
        
        // Check compression type for different size limits
        const compressionType = document.querySelector('input[name="compression_type"]:checked');
        
        if (compressionType && compressionType.value === 'server') {
            // Server-side: 50MB limit
            if (sizeMB > 50) {
                resultDiv.textContent = `File ${file.name} (${sizeMB.toFixed(2)}MB) exceeds 50MB limit for server compression.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
            
            // Server-side preset validation
            const preset = document.getElementById('server-preset')?.value;
            if (!preset || !['prepress', 'printer', 'ebook', 'screen'].includes(preset)) {
                resultDiv.textContent = 'Invalid server compression preset.';
                resultDiv.classList.add('text-red-600');
                return false;
            }
        } else {
            // Client-side: 250MB limit (or keep your existing limit)
            if (sizeMB > 250) {
                resultDiv.textContent = `File ${file.name} exceeds 250MB limit.`;
                resultDiv.classList.add('text-red-600');
                return false;
            }
            
            // Client-side doesn't need preset validation
            console.log('Client-side compression - using intelligent analysis');
        }
        
        // Common validation for both
        if (file.type !== 'application/pdf') {
            resultDiv.textContent = `File ${file.name} must be a PDF.`;
            resultDiv.classList.add('text-red-600');
            return false;
        }
        
        return true;
    }

    // } else if (endpoint === 'compress_pdf') {
    //     const file = files[0];
    //     const sizeMB = file.size / (1024 * 1024);
    //     if (sizeMB > 250) { // Increased limit for server-side
    //         resultDiv.textContent = `File ${file.name} exceeds 250MB limit.`;
    //         resultDiv.classList.add('text-red-600');
    //         return false;
    //     }
    //     if (file.type !== 'application/pdf') {
    //         resultDiv.textContent = `File ${file.name} must be a PDF.`;
    //         resultDiv.classList.add('text-red-600');
    //         return false;
    //     }

    //     // Check compression type
    //     const compressionType = document.querySelector('input[name="compression_type"]:checked');
    //     if (compressionType && compressionType.value === 'server') {
    //         // Server-side validation
    //         const preset = document.getElementById('server-preset')?.value;
    //         if (!preset || !['prepress', 'printer', 'ebook', 'screen'].includes(preset)) {
    //             resultDiv.textContent = 'Invalid server compression preset.';
    //             resultDiv.classList.add('text-red-600');
    //             return false;
    //         }
    //     } else {
    //         // Client-side validation (your existing code)
    //         const preset = form.querySelector('select[name="preset"]').value;
    //         if (!['High', 'Medium', 'Low', 'Custom'].includes(preset)) {
    //             resultDiv.textContent = 'Invalid preset. Choose High, Medium, Low, or Custom.';
    //             resultDiv.classList.add('text-red-600');
    //             return false;
    //         }
    //         if (preset === 'Custom') {
    //             const customDpi = form.querySelector('input[name="custom_dpi"]').value;
    //             const customQuality = form.querySelector('input[name="custom_quality"]').value;
    //             if (!customDpi || !customQuality) {
    //                 resultDiv.textContent = 'Custom preset requires DPI and quality values.';
    //                 resultDiv.classList.add('text-red-600');
    //                 return false;
    //             }
    //             const dpi = parseInt(customDpi);
    //             const quality = parseInt(customQuality);
    //             if (dpi < 50 || dpi > 400 || quality < 10 || quality > 100) {
    //                 resultDiv.textContent = 'Invalid custom DPI (50-400) or quality (10-100).';
    //                 resultDiv.classList.add('text-red-600');
    //                 return false;
    //             }
    //         }
    //     }
    // }

    else if (endpoint === 'delete_pdf_pages') {
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
    }


    else {
        const maxSizeMB = endpoint === 'compress_pdf' ? 55 : endpoint === 'split_pdf' ? 100 : 70;
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

    // handle raw URLs that might not be in markdown format
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

// styles to document
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

        // Send only LAST 4 conversations (8 messages) to LLM
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

        // Add BOTH user message and AI response to FULL history (for UI)
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

        // Keep full history for UI
        if (chatHistory.length > 100) { // Keep reasonable limit for browser memory
            chatHistory = chatHistory.slice(-100);
            console.log("🗂️ Trimmed full history to 100 messages for UI");
        }


        console.log("📝 Updated FULL chatHistory (UI):", chatHistory.length, "messages");
        console.log("📝 Next LLM will receive:", Math.min(chatHistory.length, 8), "messages");

    } catch (error) {
        console.error("Chat error:", error);

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


function showTool(toolId, event = null) {
    localStorage.setItem('lastTool', toolId);
    console.log("Showing tool:", toolId);

    // Hide all sections
    document.querySelectorAll('.tool-section').forEach(section => {
        section.style.display = 'none';
    });

    // Show selected section
    const toolSection = document.getElementById(toolId);
    if (toolSection) {
        toolSection.style.display = 'block';
    }


    // Show/hide clear form button
    const clearBtn = document.getElementById('clear-all-btn-container');
    if (toolId === 'chat-section') {
        clearBtn.style.display = 'none';
    } else {
        clearBtn.style.display = 'block';
    }

    // Update navigation styling
    document.querySelectorAll('.nav-link, .dropdown-content a').forEach(link => {
        link.classList.remove('text-green-600');
        link.classList.add('text-blue-600');
    });

    if (event && event.currentTarget) {
        event.currentTarget.classList.add('text-green-600');
    }

    // Mobile menu handling
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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    // Preload common libraries in background
    // pdfLibraryManager.loadLibrary('fileSaver').catch(console.warn);
    // pdfLibraryManager.loadLibrary('jszip').catch(console.warn);

    // Show last used tool or chat by default
    const lastTool = localStorage.getItem('lastTool') || 'chat-section';
    showTool(lastTool);
    setupCompressionType();
    console.log('PDF Library Manager initialized');
    console.log('Library status:', pdfLibraryManager.getStatus());
});

// 

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
        modeToggle.addEventListener('change', function () {
            modeSelect.disabled = !this.checked;
            if (this.checked && !modeSelect.value) {
                modeSelect.value = "general"; // Auto-select General mode
            }
            updateModeUI();
        });

        // Also update when mode selection changes
        modeSelect.addEventListener('change', updateModeUI);
    }


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

    const compressionResults = document.getElementById('compression-results');
    if (compressionResults) {
        const compressionSizes = document.getElementById('compression-sizes');
        if (compressionSizes) compressionSizes.innerHTML = ''; // clear list items
        compressionResults.classList.add('hidden'); // hide entire block
    }


    document.getElementById("progress-text-signatureForm").textContent = "";
    document.getElementById("result-signatureForm").textContent = "";
    document.querySelector("#signatureForm progress").value = 0;
    document.querySelector("#signatureForm progress").style.display = "none";

    // **Clear page previews of rotate page**
    const rotatePreviewContainer = document.getElementById('rotate-pages-preview-container');
    if (rotatePreviewContainer) {
        rotatePreviewContainer.innerHTML = '';
    }

    // Also reset rotation-related global variables
    currentPDFDoc = null;
    selectedPagesForRotation.clear();
    pageRotations.clear();

    //  clear form insert page in pdf
    document.getElementById('insertPdf-main-file').value = '';
    document.getElementById('insertPdf-insert-file').value = '';
    document.getElementById('insertPdf-main-file-name').textContent = 'No file selected';
    document.getElementById('insertPdf-insert-file-name').textContent = 'No file selected';
    document.getElementById('insertPdf-main-pages').textContent = 'Total Pages: Not loaded';
    document.getElementById('insertPdf-insert-pages').textContent = 'Total Pages: Not loaded';
    document.getElementById('insertPdf-previews').classList.add('hidden');
    document.getElementById('result-insertPdfForm').innerHTML = '';

    mainPDFDoc = null;
    insertPDFDoc = null;
    mainPageOrder = [];
    selectedMainPages.clear();

    // Reset select all button
    const button = document.getElementById('select-all-main-pages-btn');
    if (button) {
        button.innerHTML = '<i class="fas fa-check-square mr-2"></i> Select All';
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-gray-600', 'hover:bg-gray-700');
    }

}


//  abort compression opn\\

function stopAllOperations() {
    console.log('🛑 STOPPING ALL OPERATIONS');
    
    // For client-side: refresh page (kills everything)
    window.location.reload();
    
    // For server-side: call stop endpoint
    fetch('/stop_operations', { method: 'POST' })
        .then(() => console.log('Server operations stopped'))
        .catch(err => console.log('Stop request sent'));
}

