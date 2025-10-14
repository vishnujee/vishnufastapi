
async function convertPDFToImagesClientSide() {
    console.log('Starting client-side PDF to images conversion...');

    const form = document.getElementById('pdfToImagesForm');
    const fileInput = document.getElementById('pdfToImages-file');
    const resultDiv = document.getElementById('result-pdfToImagesForm');
    const progressDiv = document.getElementById('progress-pdfToImagesForm');
    const progressText = document.getElementById('progress-text-pdfToImagesForm');
    const submitButton = form.querySelector('button');

    // Validation
    if (!fileInput || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    if (file.size > 150 * 1024 * 1024) { alert("Use PDF less than 150 mb"); return; }
    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Loading PDF...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-file-image mr-2"></i> Converting...';

    try {
        // Load PDF document
        const pdfBytes = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: pdfBytes }).promise;
        const numPages = pdf.numPages;

        console.log(`Processing ${numPages} pages...`);

        const zip = new JSZip();
        let processedPages = 0;

        for (let pageNum = 1; pageNum <= numPages; pageNum++) {
            const progress = Math.round((pageNum / numPages) * 100);
            progressText.textContent = `Converting page ${pageNum}/${numPages}... (${progress}%)`;

            const page = await pdf.getPage(pageNum);
            const viewport = page.getViewport({ scale: 2.0 }); // Higher scale for better quality

            const canvas = document.createElement('canvas');
            canvas.width = viewport.width;
            canvas.height = viewport.height;

            const context = canvas.getContext('2d');

            // White background
            context.fillStyle = 'white';
            context.fillRect(0, 0, canvas.width, canvas.height);

            // Render PDF page
            const renderContext = {
                canvasContext: context,
                viewport: viewport
            };

            await page.render(renderContext).promise;

            // Convert to blob and add to zip
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/png', 0.9);
            });

            zip.file(`page_${pageNum}.png`, blob);
            processedPages++;

            // Clean up
            canvas.remove();
        }

        progressText.textContent = 'Creating ZIP file...';

        // Generate ZIP file
        const zipBlob = await zip.generateAsync({ type: 'blob' });

        // Download the ZIP file
        const filename = `pdf_images_${Date.now()}.zip`;
        saveAs(zipBlob, filename);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF Converted to Images Successfully!</strong><br>
                📁 File: ${filename}<br>
                📄 Pages converted: ${processedPages}<br>
            
            </div>
        `;

    } catch (error) {
        console.error('PDF to images conversion failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Conversion failed: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('convert_pdf_to_images', 'pdfToImagesForm');
        }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-file-image mr-2"></i> Convert to Images';
    }
}

async function splitPDFClientSide() {
    console.log('Starting client-side PDF split...');

    const form = document.getElementById('splitForm');
    const fileInput = document.getElementById('split-file');
    const resultDiv = document.getElementById('result-splitForm');
    const progressDiv = document.getElementById('progress-splitForm');
    const progressText = document.getElementById('progress-text-splitForm');
    const submitButton = form.querySelector('button');

    // Validation
    if (!fileInput || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }
    if (file.size > 150 * 1024 * 1024) { alert("Use PDF less than 150 mb"); return; }
    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Loading PDF...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-cut mr-2"></i> Splitting...';

    try {
        // Load PDF document
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFLib.PDFDocument.load(pdfBytes);
        const pages = pdfDoc.getPages();
        const numPages = pages.length;

        console.log(`Splitting ${numPages} pages...`);

        const zip = new JSZip();
        let processedPages = 0;

        for (let i = 0; i < numPages; i++) {
            const progress = Math.round(((i + 1) / numPages) * 100);
            progressText.textContent = `Splitting page ${i + 1}/${numPages}... (${progress}%)`;

            // Create new PDF with single page
            const singlePagePdf = await PDFLib.PDFDocument.create();
            const [copiedPage] = await singlePagePdf.copyPages(pdfDoc, [i]);
            singlePagePdf.addPage(copiedPage);

            // Save single page PDF
            const singlePageBytes = await singlePagePdf.save();
            zip.file(`page_${i + 1}.pdf`, singlePageBytes);
            processedPages++;
        }

        progressText.textContent = 'Creating ZIP file...';

        // Generate ZIP file
        const zipBlob = await zip.generateAsync({ type: 'blob' });

        // Download the ZIP file
        const filename = `split_pages_${Date.now()}.zip`;
        saveAs(zipBlob, filename);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF Split Successfully!</strong><br>
                📁 File: ${filename}<br>
                📄 Pages split: ${processedPages}<br>
               
            </div>
        `;

    } catch (error) {
        console.error('PDF split failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Split failed: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('split_pdf', 'splitForm');
        }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-cut mr-2"></i> Split PDF';
    }
}


//  PDF TO PPT

async function convertPDFToPPTClientSide() {
    console.log('Starting client-side PDF to PPT conversion...');

    const form = document.getElementById('pdfToPptForm');
    const fileInput = document.getElementById('pdfToPpt-file');
    const resultDiv = document.getElementById('result-pdfToPptForm');
    const progressDiv = document.getElementById('progress-pdfToPptForm');
    const progressText = document.getElementById('progress-text-pdfToPptForm');
    const submitButton = form.querySelector('button[type="button"]');
    const conversionType = form.querySelector('input[name="conversionType"]:checked').value;

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }
    if (file.size > 150 * 1024 * 1024) {
        alert("Use PDF less than 150 mb");
        return;
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting conversion...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-file-powerpoint mr-2"></i> Converting...';

    try {
        progressText.textContent = 'Loading PDF...';

        // Check if required libraries are available
        if (typeof PDFLib === 'undefined') {
            throw new Error('PDF library not loaded. Please refresh the page.');
        }

        const { PDFDocument } = PDFLib;

        // Load PDF and create a copy for PDF.js
        const pdfBytes = await file.arrayBuffer();

        // Create a copy of the ArrayBuffer for PDF.js to prevent detachment issues
        const pdfBytesCopy = pdfBytes.slice(0);

        const pdfDoc = await PDFDocument.load(pdfBytes);
        const numPages = pdfDoc.getPageCount();

        console.log(`Processing ${numPages} pages for PPT conversion...`);
        if (conversionType === 'image') {
            progressText.textContent = 'Converting pages to images...';

            const pptx = new PptxGenJS();
            pptx.layout = 'LAYOUT_WIDE';

            const pdfjsDoc = await pdfjsLib.getDocument({ data: pdfBytesCopy }).promise;

            for (let i = 0; i < numPages; i++) {
                const progress = Math.round((i / numPages) * 80);
                progressText.textContent = `Processing page ${i + 1}/${numPages}... (${progress}%)`;

                const page = pdfDoc.getPage(i);
                const { width, height } = page.getSize();

                // Adaptive scaling based on page size
                const maxDimension = width > 2000 || height > 2000 ? 800 : 1200;
                const scale = Math.min(maxDimension / Math.max(width, height), 2);

                const canvas = document.createElement('canvas');
                canvas.width = Math.floor(width * scale);
                canvas.height = Math.floor(height * scale);

                const context = canvas.getContext('2d');
                context.fillStyle = 'white';
                context.fillRect(0, 0, canvas.width, canvas.height);

                const pdfjsPage = await pdfjsDoc.getPage(i + 1);
                const viewport = pdfjsPage.getViewport({ scale: scale });

                await pdfjsPage.render({
                    canvasContext: context,
                    viewport: viewport
                }).promise;

                // Adaptive quality based on content
                const hasComplexGraphics = canvas.width > 1000 || canvas.height > 1000;
                const quality = hasComplexGraphics ? 0.7 : 0.8;
                const format = hasComplexGraphics ? 'image/jpeg' : 'image/png';

                const imageData = canvas.toDataURL(format, quality);

                const slide = pptx.addSlide();

                // Fit to slide with proper margins
                const aspectRatio = width / height;
                // const maxWidth = 9.5;
                // const maxHeight = 6.5;
                // const maxWidth = 9.8;  // Increased from 9.5
                // const maxHeight = 7.0; // Increased from 6.5
                const maxWidth = 9.9;  // Almost full width
                const maxHeight = 7.4; // Almost full height

                let imgWidth, imgHeight;
                if (aspectRatio > maxWidth / maxHeight) {
                    imgWidth = maxWidth;
                    imgHeight = maxWidth / aspectRatio;
                } else {
                    imgHeight = maxHeight;
                    imgWidth = maxHeight * aspectRatio;
                }

                const xOffset = (10 - imgWidth) / 2 + 1.4;
                const yOffset = (7.5 - imgHeight) / 2;

                slide.addImage({
                    data: imageData,
                    x: xOffset,
                    y: yOffset,
                    w: imgWidth,
                    h: imgHeight,
                    sizing: { type: 'contain' }
                });

                canvas.remove();
            }

            progressText.textContent = 'Generating PowerPoint file...';
            await pptx.writeFile({
                fileName: `converted_${file.name.replace('.pdf', '')}.pptx`
            });
        }

        else {
            // Editable text conversion (basic implementation)
            throw new Error('Editable conversion requires server processing. Please use image-based conversion for client-side.');
        }

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF to PowerPoint Conversion Successful!</strong><br>
                📊 Converted ${numPages} pages to PowerPoint<br>
            </div>
        `;

    } catch (error) {
        console.error('PDF to PPT conversion failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Conversion failed: ${error.message}<br>
                <small>Falling back to other processing...</small>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('convert_pdf_to_ppt', 'pdfToPptForm');
        }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-file-powerpoint mr-2"></i> Convert to PowerPoint';
    }
}

// async function convertPDFToPPTClientSide() {
//     console.log('Starting client-side PDF to PPT conversion...');

//     const form = document.getElementById('pdfToPptForm');
//     const fileInput = document.getElementById('pdfToPpt-file');
//     const resultDiv = document.getElementById('result-pdfToPptForm');
//     const progressDiv = document.getElementById('progress-pdfToPptForm');
//     const progressText = document.getElementById('progress-text-pdfToPptForm');
//     const submitButton = form.querySelector('button[type="button"]');
//     const conversionType = form.querySelector('input[name="conversionType"]:checked').value;

//     // Validation
//     if (!fileInput || !fileInput.files || !fileInput.files[0]) {
//         alert('Please select a PDF file.');
//         return;
//     }

//     const file = fileInput.files[0];

//     // Validate file type
//     if (file.type !== 'application/pdf') {
//         alert('Please select a PDF file.');
//         return;
//     }
//     if (file.size > 150 * 1024 * 1024) { alert("Use PDF less than 150 mb"); return; }
//     // Show progress
//     progressDiv.style.display = 'block';
//     progressText.textContent = 'Starting conversion...';
//     submitButton.disabled = true;
//     submitButton.innerHTML = '<i class="fas fa-file-powerpoint mr-2"></i> Converting...';

//     try {
//         progressText.textContent = 'Loading PDF...';

//         // Check if required libraries are available
//         if (typeof PDFLib === 'undefined') {
//             throw new Error('PDF library not loaded. Please refresh the page.');
//         }

//         const { PDFDocument } = PDFLib;

//         // Load PDF
//         const pdfBytes = await file.arrayBuffer();
//         const pdfDoc = await PDFDocument.load(pdfBytes);
//         const numPages = pdfDoc.getPageCount();

//         console.log(`Processing ${numPages} pages for PPT conversion...`);

//         if (conversionType === 'image') {
//             // Image-based conversion (pages as images in PPT)
//             progressText.textContent = 'Converting pages to images...';

//             const pptx = new PptxGenJS();

//             for (let i = 0; i < numPages; i++) {
//                 const progress = Math.round((i / numPages) * 80);
//                 progressText.textContent = `Processing page ${i + 1}/${numPages}... (${progress}%)`;

//                 const page = pdfDoc.getPage(i);
//                 const { width, height } = page.getSize();

//                 // Create canvas for rendering
//                 const canvas = document.createElement('canvas');
//                 const scale = 2; // Higher resolution for better quality
//                 canvas.width = width * scale;
//                 canvas.height = height * scale;

//                 const context = canvas.getContext('2d');
//                 context.fillStyle = 'white';
//                 context.fillRect(0, 0, canvas.width, canvas.height);

//                 // Render PDF page to canvas
//                 const pdfjsPage = await pdfjsLib.getDocument({ data: pdfBytes }).promise.then(pdf => pdf.getPage(i + 1));
//                 const viewport = pdfjsPage.getViewport({ scale });

//                 await pdfjsPage.render({
//                     canvasContext: context,
//                     viewport: viewport
//                 }).promise;

//                 // Convert canvas to image
//                 const imageData = canvas.toDataURL('image/png', 0.8);

//                 // Add slide with image
//                 const slide = pptx.addSlide();
//                 slide.addImage({
//                     data: imageData,
//                     x: 0.5,
//                     y: 0.5,
//                     w: 9,
//                     h: 6,
//                     sizing: { type: 'contain', w: 9, h: 6 }
//                 });

//                 // Clean up
//                 canvas.remove();
//             }

//             progressText.textContent = 'Generating PowerPoint file...';

//             // Generate and download PPT
//             const pptBlob = await pptx.writeFile({ outputType: 'blob' });
//             const url = URL.createObjectURL(pptBlob);
//             const a = document.createElement('a');
//             a.href = url;
//             a.download = `converted_${file.name.replace('.pdf', '')}.pptx`;
//             document.body.appendChild(a);
//             a.click();
//             document.body.removeChild(a);
//             URL.revokeObjectURL(url);

//         } else {
//             // Editable text conversion (basic implementation)
//             throw new Error('Editable conversion requires server processing. Please use image-based conversion for client-side.');
//         }

//         resultDiv.innerHTML = `
//             <div class="text-green-600">
//                 ✅ <strong>PDF to PowerPoint Conversion Successful!</strong><br>
//                 📊 Converted ${numPages} pages to PowerPoint<br>

//             </div>
//         `;

//     } catch (error) {
//         console.error('PDF to PPT conversion failed:', error);

//         resultDiv.innerHTML = `
//             <div class="text-red-600">
//                 ❌ Conversion failed: ${error.message}<br>
//                 <small>Falling back to server processing...</small>
//             </div>
//         `;

//         // Fallback to server processing
//         setTimeout(() => {
//             processPDF('convert_pdf_to_ppt', 'pdfToPptForm');
//         }, 2000);

//     } finally {
//         progressDiv.style.display = 'none';
//         submitButton.disabled = false;
//         submitButton.innerHTML = '<i class="fas fa-file-powerpoint mr-2"></i> Convert to PowerPoint';
//     }
// }


// IMAGE TO PDF

async function convertImageToPDFClientSide() {
    console.log('Starting client-side Image to PDF conversion...');

    const form = document.getElementById('imageToPdfForm');
    const fileInput = document.getElementById('imageToPdf-file');
    const resultDiv = document.getElementById('result-imageToPdfForm');
    const progressDiv = document.getElementById('progress-imageToPdfForm');
    const progressText = document.getElementById('progress-text-imageToPdfForm');

    // Get form values
    const description = document.getElementById('image-description')?.value || '';
    const descriptionPosition = document.getElementById('description-position')?.value || 'bottom-center';
    const fontSize = parseInt(document.getElementById('description-font-size')?.value) || 20;
    const pageSize = document.getElementById('page_size')?.value || 'A4';
    const orientation = document.getElementById('orientation')?.value || 'Portrait';
    const fontColor = document.getElementById('font-color')?.value || '#000000';
    const fontFamily = document.getElementById('font-family')?.value || 'helvetica';
    const fontWeight = document.getElementById('font-weight')?.value || 'normal';
    const customX = document.getElementById('custom-x')?.value;
    const customY = document.getElementById('custom-y')?.value;

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        alert('Please select an image file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
        alert('Please select a PNG or JPEG image file.');
        return;
    }

    if (file.size > 150 * 1024 * 1024) { alert("Use PDF less than 150 mb"); return; }
    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting conversion...';

    try {
        progressText.textContent = 'Processing image...';

        // Check if required libraries are available
        if (typeof PDFLib === 'undefined') {
            throw new Error('PDF library not loaded. Please refresh the page.');
        }

        const { PDFDocument, rgb } = PDFLib;

        // Create new PDF document
        const pdfDoc = await PDFDocument.create();

        // Set page size
        const pageDimensions = getPageDimensions(pageSize, orientation);

        // Add a page
        const page = pdfDoc.addPage([pageDimensions.width, pageDimensions.height]);

        // Load and embed image
        // const imageBytes = await file.arrayBuffer();
        // let image;

        // if (file.type === 'image/png') {
        //     image = await pdfDoc.embedPng(imageBytes);
        // } else {
        //     image = await pdfDoc.embedJpg(imageBytes);
        // }

        // FIX: Load image through canvas to fix orientation
        const correctedBytes = await loadAndFixImageOrientation(file);
        let image;

        if (file.type === 'image/png') {
            image = await pdfDoc.embedPng(correctedBytes);
        } else {
            image = await pdfDoc.embedJpg(correctedBytes);
        }

        // Add this helper function outside convertImageToPDFClientSide():
        async function loadAndFixImageOrientation(file) {
            return new Promise((resolve) => {
                const img = new Image();
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                img.onload = function () {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);

                    canvas.toBlob(function (blob) {
                        blob.arrayBuffer().then(resolve);
                    }, file.type);
                };
                img.src = URL.createObjectURL(file);
            });
        }

        // Calculate image dimensions to fit page with margins
        const margin = 50;
        const maxWidth = pageDimensions.width - (2 * margin);
        const maxHeight = pageDimensions.height - (2 * margin);

        const imageDims = image.scaleToFit(maxWidth, maxHeight);
        const x = margin + (maxWidth - imageDims.width) / 2;
        const y = margin + (maxHeight - imageDims.height) / 2;

        // Draw image
        page.drawImage(image, {
            x,
            y,
            width: imageDims.width,
            height: imageDims.height,
        });

        // Add description if provided
        if (description.trim()) {
            progressText.textContent = 'Adding description...';

            // Convert hex color to RGB
            const hexToRgb = (hex) => {
                const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                return result ? {
                    r: parseInt(result[1], 16) / 255,
                    g: parseInt(result[2], 16) / 255,
                    b: parseInt(result[3], 16) / 255
                } : { r: 0, g: 0, b: 0 };
            };

            const color = hexToRgb(fontColor);

            // Get the appropriate font
            let font;
            switch (fontFamily) {
                case 'times':
                    font = pdfDoc.embedStandardFont('TimesRoman');
                    break;
                case 'courier':
                    font = pdfDoc.embedStandardFont('Courier');
                    break;
                case 'zapf':
                    font = pdfDoc.embedStandardFont('ZapfDingbats');
                    break;
                case 'helvetica':
                default:
                    font = fontWeight === 'bold' ? pdfDoc.embedStandardFont('Helvetica-Bold') : pdfDoc.embedStandardFont('Helvetica');
                    break;
            }

            // Calculate text width for proper positioning
            const textWidth = font.widthOfTextAtSize(description, fontSize);

            // Calculate text position - FIXED LOGIC
            let textX, textY;
            const textMargin = 30;

            if (descriptionPosition === 'custom' && customX && customY) {
                textX = parseFloat(customX);
                textY = parseFloat(customY);
            } else {
                switch (descriptionPosition) {
                    case 'top':
                        textX = pageDimensions.width / 2;
                        textY = pageDimensions.height - textMargin;
                        break;
                    case 'top-center':
                        textX = (pageDimensions.width - textWidth) / 2;
                        textY = pageDimensions.height - textMargin - fontSize; // Subtract font size
                        break;
                    case 'top-left':
                        textX = textMargin;
                        textY = pageDimensions.height - textMargin;
                        break;
                    case 'top-right':
                        textX = pageDimensions.width - textWidth - textMargin;
                        textY = pageDimensions.height - textMargin;
                        break;
                    case 'bottom':
                        textX = pageDimensions.width / 2;
                        textY = textMargin;
                        break;
                    case 'bottom-center':
                        textX = (pageDimensions.width - textWidth) / 2;
                        textY = textMargin; // Keep at bottom margin
                        break;
                    case 'bottom-left':
                        textX = textMargin;
                        textY = textMargin + fontSize;
                        break;
                    case 'bottom-right':
                        textX = pageDimensions.width - textWidth - textMargin;
                        textY = textMargin + fontSize;
                        break;
                    default:
                        textX = (pageDimensions.width - textWidth) / 2;
                        textY = textMargin + fontSize;
                        break;
                }
            }

            console.log(`Text position: ${descriptionPosition}, X: ${textX}, Y: ${textY}, Width: ${textWidth}`);

            // Draw description text with proper alignment
            page.drawText(description, {
                x: textX,
                y: textY,
                size: fontSize,
                color: rgb(color.r, color.g, color.b),
                font: font,
                lineHeight: fontSize * 1.2,
                maxWidth: pageDimensions.width - (2 * textMargin),
            });
        }

        progressText.textContent = 'Generating PDF...';

        // Save PDF
        const pdfBytes = await pdfDoc.save();
        const pdfBlob = new Blob([pdfBytes], { type: 'application/pdf' });

        // Download
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `converted_${file.name.replace(/\.[^/.]+$/, "")}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Image to PDF Conversion Successful!</strong><br>
                🖼️ Converted "${file.name}" to PDF<br>
                📍 Description Position: ${descriptionPosition}<br>
              
            </div>
        `;

    } catch (error) {
        console.error('Image to PDF conversion failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Conversion failed: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('convert_image_to_pdf', 'imageToPdfForm');
        }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        progressText.textContent = '';
    }
}

function getPageDimensions(pageSize, orientation) {
    const sizes = {
        'A4': { width: 595, height: 842 },
        'Letter': { width: 612, height: 792 }
    };

    const size = sizes[pageSize] || sizes['A4'];
    return orientation === 'Landscape'
        ? { width: size.height, height: size.width }
        : size;
}




//  ADD PAGE NUMBER

async function addPageNumbersClientSide() {
    console.log('Starting client-side page numbering...');

    const form = document.getElementById('pageNumbersForm');
    const fileInput = document.getElementById('pageNumbers-file');
    const positionSelect = document.getElementById('pageNumbers-position');
    const alignmentSelect = document.getElementById('pageNumbers-alignment');
    const formatSelect = document.getElementById('pageNumbers-format');
    const resultDiv = document.getElementById('result-pageNumbersForm');
    const progressDiv = document.getElementById('progress-pageNumbersForm');
    const progressText = document.getElementById('progress-text-pageNumbersForm');
    const submitButton = form.querySelector('button[type="button"]');

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    // Validate file type
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }
    if (file.size > 150 * 1024 * 1024) { alert("Use PDF less than 150 mb"); return; }

    // Get form values
    const position = positionSelect ? positionSelect.value : 'bottom';
    const alignment = alignmentSelect ? alignmentSelect.value : 'center';
    const format = formatSelect ? formatSelect.value : 'page_x';

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting page numbering...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-list-ol mr-2"></i> Adding Page Numbers...';

    try {
        progressText.textContent = 'Loading PDF...';

        // Check if required libraries are available
        if (typeof PDFLib === 'undefined') {
            throw new Error('PDF library not loaded. Please refresh the page.');
        }

        const { PDFDocument, rgb, StandardFonts } = PDFLib;

        // Load PDF
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFDocument.load(pdfBytes);
        const numPages = pdfDoc.getPageCount();

        console.log(`Adding page numbers to ${numPages} pages...`);

        // Embed font
        const font = await pdfDoc.embedFont(StandardFonts.Helvetica);

        // Process each page
        for (let i = 0; i < numPages; i++) {
            const progress = Math.round((i / numPages) * 90);
            progressText.textContent = `Processing page ${i + 1}/${numPages}... (${progress}%)`;

            const page = pdfDoc.getPage(i);
            const { width, height } = page.getSize();

            // Generate page number text
            const pageNumber = i + 1;
            const pageText = format === 'page_x' ? `Page ${pageNumber}` : `${pageNumber}`;

            // Calculate position
            const margin = 30; // Distance from edge
            const fontSize = 12;

            let x, y;

            // Vertical position
            switch (position) {
                case 'top':
                    y = height - margin;
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
                case 'right':
                    x = width - margin - font.widthOfTextAtSize(pageText, fontSize);
                    break;
                case 'center':
                default:
                    x = (width - font.widthOfTextAtSize(pageText, fontSize)) / 2;
                    break;
            }

            // Add page number
            page.drawText(pageText, {
                x,
                y,
                size: fontSize,
                font: font,
                color: rgb(0, 0, 0), // Black color
            });

            console.log(`Added page number "${pageText}" to page ${pageNumber} at (${x}, ${y})`);
        }

        progressText.textContent = 'Saving PDF with page numbers...';

        // Save the modified PDF
        const pdfWithNumbers = await pdfDoc.save();
        const pdfBlob = new Blob([pdfWithNumbers], { type: 'application/pdf' });

        // Download
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `numbered_${file.name}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Page Numbers Added Successfully!</strong><br>
                📄 Added numbers to ${numPages} pages<br>
                📍 Position: ${position}, Alignment: ${alignment}<br>
                
            </div>
        `;

        console.log(`Successfully added page numbers to ${numPages} pages`);

    } catch (error) {
        console.error('Page numbering failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Failed to add page numbers: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('add_page_numbers', 'pageNumbersForm');
        }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-list-ol mr-2"></i> Add Page Numbers';
    }
}

// Enhanced version with more customization options
async function addPageNumbersAdvancedClientSide() {
    console.log('Starting advanced client-side page numbering...');

    const form = document.getElementById('pageNumbersForm');
    const fileInput = document.getElementById('pageNumbers-file');
    const positionSelect = document.getElementById('pageNumbers-position');
    const alignmentSelect = document.getElementById('pageNumbers-alignment');
    const formatSelect = document.getElementById('pageNumbers-format');
    const resultDiv = document.getElementById('result-pageNumbersForm');
    const progressDiv = document.getElementById('progress-pageNumbersForm');
    const progressText = document.getElementById('progress-text-pageNumbersForm');
    const submitButton = form.querySelector('button[type="button"]');

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    // Get form values
    const position = positionSelect ? positionSelect.value : 'bottom';
    const alignment = alignmentSelect ? alignmentSelect.value : 'center';
    const format = formatSelect ? formatSelect.value : 'page_x';

    // Advanced options (you can add these to your HTML form later)
    const fontSize = 12;
    const fontColor = '#000000'; // Black
    const startFrom = 1; // Starting page number
    const excludeFirstPage = false; // Option to exclude cover page

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting advanced page numbering...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-list-ol mr-2"></i> Adding Page Numbers...';

    try {
        progressText.textContent = 'Loading PDF...';

        if (typeof PDFLib === 'undefined') {
            throw new Error('PDF library not loaded. Please refresh the page.');
        }

        const { PDFDocument, rgb, StandardFonts } = PDFLib;

        // Load PDF
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFDocument.load(pdfBytes);
        const numPages = pdfDoc.getPageCount();

        console.log(`Adding advanced page numbers to ${numPages} pages...`);

        // Embed fonts
        const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
        const boldFont = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

        // Convert hex color to RGB
        const hexToRgb = (hex) => {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16) / 255,
                g: parseInt(result[2], 16) / 255,
                b: parseInt(result[3], 16) / 255
            } : { r: 0, g: 0, b: 0 };
        };

        const color = hexToRgb(fontColor);

        // Process each page
        for (let i = 0; i < numPages; i++) {
            const progress = Math.round((i / numPages) * 90);
            progressText.textContent = `Processing page ${i + 1}/${numPages}... (${progress}%)`;

            // Skip first page if excludeFirstPage is true
            if (excludeFirstPage && i === 0) {
                continue;
            }

            const page = pdfDoc.getPage(i);
            const { width, height } = page.getSize();

            // Calculate page number (adjust for startFrom and exclusions)
            const pageNumber = startFrom + (excludeFirstPage ? i - 1 : i);

            // Generate page number text based on format
            let pageText;
            switch (format) {
                case 'x':
                    pageText = `${pageNumber}`;
                    break;
                case 'page_x_of_y':
                    pageText = `Page ${pageNumber} of ${numPages - (excludeFirstPage ? 1 : 0)}`;
                    break;
                case 'page_x':
                default:
                    pageText = `Page ${pageNumber}`;
                    break;
            }

            // Calculate position with margins
            const margin = 40;
            const textWidth = font.widthOfTextAtSize(pageText, fontSize);

            let x, y;

            // Vertical position with offset
            switch (position) {
                case 'top':
                    y = height - margin;
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
                case 'right':
                    x = width - margin - textWidth;
                    break;
                case 'center':
                default:
                    x = (width - textWidth) / 2;
                    break;
            }

            // Add background rectangle for better visibility (optional)
            const bgPadding = 2;
            page.drawRectangle({
                x: x - bgPadding,
                y: y - bgPadding,
                width: textWidth + (2 * bgPadding),
                height: fontSize + (2 * bgPadding),
                color: rgb(1, 1, 1), // White background
                opacity: 0.8, // Semi-transparent
            });

            // Add page number text
            page.drawText(pageText, {
                x,
                y,
                size: fontSize,
                font: font,
                color: rgb(color.r, color.g, color.b),
            });

            console.log(`Added page number "${pageText}" to page ${i + 1}`);
        }

        progressText.textContent = 'Finalizing PDF...';

        // Save the modified PDF
        const pdfWithNumbers = await pdfDoc.save();
        const pdfBlob = new Blob([pdfWithNumbers], { type: 'application/pdf' });

        // Download
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `numbered_${file.name}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        const pagesProcessed = excludeFirstPage ? numPages - 1 : numPages;

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Advanced Page Numbers Added Successfully!</strong><br>
                📄 Processed ${pagesProcessed} pages<br>
                📍 Position: ${position}, Alignment: ${alignment}, Format: ${format}<br>
                🎨 Font Size: ${fontSize}pt, Starting from: ${startFrom}<br>
                
            </div>
        `;

        console.log(`Successfully added advanced page numbers to ${pagesProcessed} pages`);

    } catch (error) {
        console.error('Advanced page numbering failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Failed to add page numbers: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('add_page_numbers', 'pageNumbersForm');
        }, 2000);

    } finally {
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-list-ol mr-2"></i> Add Page Numbers';
    }
}



//  reorder page

// 🆕 FIXED: Page preview loading without duplicates
async function loadPagePreviewsClientSide(file, pageListElement) {
    if (!file || typeof pdfjsLib === 'undefined') {
        throw new Error('PDF library not available');
    }

    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const numPages = pdf.numPages;

        pageListElement.innerHTML = '';

        console.log(`Loading ${numPages} page previews...`);

        for (let i = 1; i <= numPages; i++) {
            const page = await pdf.getPage(i);
            const viewport = page.getViewport({ scale: 0.3 });

            // Create only one canvas - the one that will be in the DOM
            const canvas = document.createElement('canvas');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            const context = canvas.getContext('2d');

            // White background
            context.fillStyle = 'white';
            context.fillRect(0, 0, canvas.width, canvas.height);

            // Render PDF page directly to the canvas
            await page.render({
                canvasContext: context,
                viewport: viewport
            }).promise;

            const pageDiv = document.createElement('div');
            pageDiv.className = 'page-preview border border-gray-200 rounded-lg p-4 bg-white mb-4 cursor-move';
            pageDiv.dataset.pageNum = i;
            pageDiv.draggable = true;

            pageDiv.innerHTML = `
                <div class="flex flex-col items-center">
                    <div class="canvas-container mb-2 border border-gray-300 bg-white overflow-hidden rounded">
                        <!-- Canvas will be appended here -->
                    </div>
                    <div class="flex items-center justify-between w-full mt-2">
                        <span class="text-gray-600 text-sm font-medium">Page ${i}</span>
                        <div class="flex space-x-1">
                            <button type="button" class="move-up bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700 transition-colors">
                                <i class="fas fa-arrow-up"></i>
                            </button>
                            <button type="button" class="move-down bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700 transition-colors">
                                <i class="fas fa-arrow-down"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;

            // Append the actual canvas to the container
            const canvasContainer = pageDiv.querySelector('.canvas-container');
            canvasContainer.appendChild(canvas);

            pageListElement.appendChild(pageDiv);
        }

        console.log(`Successfully loaded ${numPages} page previews`);
        return numPages;

    } catch (error) {
        console.error('Error loading page previews:', error);
        throw error;
    }
}

// 🆕 FIXED: Enhanced reorder function with better validation
async function reorderPDFPagesClientSide() {
    console.log('Starting client-side PDF page reordering...');

    const form = document.getElementById('reorderForm');
    const fileInput = document.getElementById('reorder-file');
    const resultDiv = document.getElementById('result-reorderForm');
    const progressDiv = document.getElementById('progress-reorderForm');
    const progressText = document.getElementById('progress-text-reorderForm');
    const pageOrderInput = document.getElementById('reorder-page-order');
    // const submitButton = form.querySelector('button[type="button"]');
    const submitButton = document.getElementById('reorder-submit-btn');

    // Validation
    if (!fileInput || !fileInput.files || !fileInput.files.length) {
        alert('Please select a PDF file.');
        return;
    }

    const file = fileInput.files[0];

    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        alert('File size exceeds 50MB limit for client-side processing.');
        return;
    }

    if (!pageOrderInput || !pageOrderInput.value) {
        alert('Please reorder pages before processing.');
        return;
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting page reordering...';
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-sort-numeric-up mr-2"></i> Reordering...';

    try {
        // Parse and validate page order
        const pageOrder = pageOrderInput.value.split(',').map(p => parseInt(p.trim()));
        console.log('Requested page order:', pageOrder);

        progressText.textContent = 'Loading PDF document...';

        // Load PDF to get actual page count
        const { PDFDocument } = PDFLib;
        const pdfBytes = await file.arrayBuffer();
        const pdfDoc = await PDFDocument.load(pdfBytes);

        const totalPages = pdfDoc.getPageCount();
        console.log('Total pages in PDF:', totalPages);

        // Validate page order
        if (pageOrder.length !== totalPages) {
            throw new Error(`Page order has ${pageOrder.length} pages but PDF has ${totalPages} pages.`);
        }

        // Check for valid page numbers and duplicates
        const seenPages = new Set();
        for (const pageNum of pageOrder) {
            if (pageNum < 1 || pageNum > totalPages) {
                throw new Error(`Invalid page number: ${pageNum}. Must be between 1 and ${totalPages}.`);
            }
            if (seenPages.has(pageNum)) {
                throw new Error(`Duplicate page number found: ${pageNum}. Each page must appear exactly once.`);
            }
            seenPages.add(pageNum);
        }

        // Create new PDF document
        const newPdfDoc = await PDFDocument.create();

        progressText.textContent = 'Reordering pages... (0%)';

        // Copy pages in the specified order
        for (let i = 0; i < pageOrder.length; i++) {
            const progress = Math.round((i / pageOrder.length) * 90);
            progressText.textContent = `Reordering pages... (${progress}%)`;

            const originalPageIndex = pageOrder[i] - 1; // Convert to 0-based index

            // Copy the page
            const [copiedPage] = await newPdfDoc.copyPages(pdfDoc, [originalPageIndex]);
            newPdfDoc.addPage(copiedPage);
        }

        progressText.textContent = 'Finalizing PDF... (95%)';

        // Save the reordered PDF
        const reorderedPdfBytes = await newPdfDoc.save();
        const reorderedBlob = new Blob([reorderedPdfBytes], { type: 'application/pdf' });

        progressText.textContent = 'Downloading... (100%)';

        // Download the reordered PDF
        const filename = `reordered_${file.name.replace('.pdf', '')}.pdf`;
        const url = URL.createObjectURL(reorderedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Show success message
        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>PDF Pages Reordered Successfully!</strong><br>
                📄 Pages reordered: ${pageOrder.join(' → ')}<br>
                <small>Original order: ${Array.from({ length: totalPages }, (_, i) => i + 1).join(', ')}</small>
            </div>
        `;

        console.log('Client-side page reordering completed successfully');

    } catch (error) {
        console.error('Client-side page reordering failed:', error);

        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Page reordering failed: ${error.message}<br>
                <small>Falling back to server processing...</small>
            </div>
        `;

        // Fallback to server processing
        setTimeout(() => {
            processPDF('reorder_pages', 'reorderForm');
        }, 2000);

    } finally {
        // Clean up
        progressDiv.style.display = 'none';
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-sort-numeric-up mr-2"></i> Reorder and Download ';
    }
}

// 🆕 FIXED: Update page order function
function updatePageOrder(pageElements) {
    const pageOrder = [];

    pageElements.forEach(pageElement => {
        const pageNum = parseInt(pageElement.dataset.pageNum);
        pageOrder.push(pageNum);
    });

    const pageOrderInput = document.getElementById('reorder-page-order');
    if (pageOrderInput) {
        pageOrderInput.value = pageOrder.join(',');
        console.log("Updated page order:", pageOrderInput.value);
    }
}

// 🆕 FIXED: Initialize drag and drop functionality
function initializeDragAndDrop() {
    const pageList = document.getElementById('page-list');

    pageList.addEventListener('dragstart', (e) => {
        if (e.target.classList.contains('page-preview')) {
            e.dataTransfer.setData('text/plain', e.target.dataset.pageNum);
            e.target.classList.add('opacity-50');
        }
    });

    pageList.addEventListener('dragend', (e) => {
        if (e.target.classList.contains('page-preview')) {
            e.target.classList.remove('opacity-50');
        }
    });

    pageList.addEventListener('dragover', (e) => {
        e.preventDefault();
        const draggable = document.querySelector('.page-preview.dragging');
        if (!draggable) return;

        const afterElement = getDragAfterElement(pageList, e.clientY);
        const currentElement = e.target.closest('.page-preview');

        if (afterElement == null) {
            pageList.appendChild(draggable);
        } else {
            pageList.insertBefore(draggable, afterElement);
        }
    });

    function getDragAfterElement(container, y) {
        const draggableElements = [...container.querySelectorAll('.page-preview:not(.dragging)')];

        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;

            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }
}




document.addEventListener('DOMContentLoaded', function () {

    // Initialize drag and drop for reordering
    initializeDragAndDrop();
    console.log('PDF tools initialized');

});