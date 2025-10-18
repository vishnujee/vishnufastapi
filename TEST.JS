
IMAGE TO PDF

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

// Add this missing function
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

async function convertMultipleImagesToPDFClientSide() {
    console.log('Starting client-side Multiple Images to PDF conversion...');
    
    const fileInput = document.getElementById('multipleImageToPdf-files');
    const resultDiv = document.getElementById('result-multipleImageToPdfForm');
    const progressDiv = document.getElementById('progress-multipleImageToPdfForm');
    const progressText = document.getElementById('progress-text-multipleImageToPdfForm');

    // Get form values
    const description = document.getElementById('multiple-image-description')?.value || '';
    const descriptionPosition = document.getElementById('multiple-description-position')?.value || 'bottom-center';
    const fontSize = parseInt(document.getElementById('multiple-description-font-size')?.value) || 20;
    const pageSize = document.getElementById('multiple-page_size')?.value || 'A4';
    const orientation = document.getElementById('multiple-orientation')?.value || 'Portrait';
    const fontColor = document.getElementById('multiple-font-color')?.value || '#000000';
    const fontFamily = document.getElementById('multiple-font-family')?.value || 'helvetica';
    const fontWeight = document.getElementById('multiple-font-weight')?.value || 'normal';
    const imagesPerPage = parseInt(document.getElementById('image-per-page')?.value) || 1;

    // Validation
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert('Please select at least one image file.');
        return;
    }

    const files = Array.from(fileInput.files);
    
    // Validate file types and sizes
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    for (const file of files) {
        if (!allowedTypes.includes(file.type)) {
            alert(`Please select only PNG or JPEG image files. Invalid file: ${file.name}`);
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            alert(`File ${file.name} is too large. Maximum size is 50MB.`);
            return;
        }
    }

    // Show progress
    progressDiv.style.display = 'block';
    progressText.textContent = 'Starting conversion...';

    try {
        progressText.textContent = 'Loading PDF library...';
        const [PDFLib] = await pdfLibraryManager.loadLibraries(['pdfLib']);

        const { PDFDocument, rgb } = PDFLib;

        // Create new PDF document
        const pdfDoc = await PDFDocument.create();

        // Set page size
        const pageDimensions = getPageDimensions(pageSize, orientation);

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

        let currentPage = null;
        let imagesOnCurrentPage = 0;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            
            progressText.textContent = `Processing image ${i + 1} of ${files.length}...`;
            
            // Create new page if needed
            if (currentPage === null || imagesOnCurrentPage >= imagesPerPage) {
                currentPage = pdfDoc.addPage([pageDimensions.width, pageDimensions.height]);
                imagesOnCurrentPage = 0;
            }

            // Load and fix image orientation
            const correctedBytes = await loadAndFixImageOrientation(file);
            let image;

            if (file.type === 'image/png') {
                image = await pdfDoc.embedPng(correctedBytes);
            } else {
                image = await pdfDoc.embedJpg(correctedBytes);
            }

            // Calculate layout based on images per page
            const imagesPerRow = imagesPerPage === 1 ? 1 : 2;
            const rows = Math.ceil(imagesPerPage / imagesPerRow);
            
            const margin = 50;
            const horizontalSpacing = 20;
            const verticalSpacing = description ? 60 : 20;
            
            const availableWidth = pageDimensions.width - (2 * margin) - ((imagesPerRow - 1) * horizontalSpacing);
            const availableHeight = pageDimensions.height - (2 * margin) - ((rows - 1) * verticalSpacing);
            
            const imageWidth = availableWidth / imagesPerRow;
            const imageHeight = availableHeight / rows;

            // Calculate position for current image
            const rowIndex = Math.floor(imagesOnCurrentPage / imagesPerRow);
            const colIndex = imagesOnCurrentPage % imagesPerRow;
            
            const x = margin + (colIndex * (imageWidth + horizontalSpacing));
            const y = pageDimensions.height - margin - ((rowIndex + 1) * imageHeight) + (rowIndex * verticalSpacing);

            // Scale image to fit the allocated space
            const imageDims = image.scaleToFit(imageWidth, imageHeight);
            const centeredX = x + (imageWidth - imageDims.width) / 2;
            const centeredY = y + (imageHeight - imageDims.height) / 2;

            // Draw image
            currentPage.drawImage(image, {
                x: centeredX,
                y: centeredY,
                width: imageDims.width,
                height: imageDims.height,
            });

            // Add description if provided
            if (description.trim()) {
                const textWidth = font.widthOfTextAtSize(description, fontSize);
                const textMargin = 10;
                
                let textX, textY;
                
                switch (descriptionPosition) {
                    case 'top':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + imageHeight - textMargin;
                        break;
                    case 'top-center':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + imageHeight - textMargin - fontSize;
                        break;
                    case 'top-left':
                        textX = x + textMargin;
                        textY = y + imageHeight - textMargin;
                        break;
                    case 'top-right':
                        textX = x + imageWidth - textWidth - textMargin;
                        textY = y + imageHeight - textMargin;
                        break;
                    case 'bottom':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + textMargin + fontSize;
                        break;
                    case 'bottom-center':
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + textMargin;
                        break;
                    case 'bottom-left':
                        textX = x + textMargin;
                        textY = y + textMargin + fontSize;
                        break;
                    case 'bottom-right':
                        textX = x + imageWidth - textWidth - textMargin;
                        textY = y + textMargin + fontSize;
                        break;
                    default:
                        textX = x + (imageWidth - textWidth) / 2;
                        textY = y + textMargin;
                        break;
                }

                currentPage.drawText(description, {
                    x: textX,
                    y: textY,
                    size: fontSize,
                    color: rgb(color.r, color.g, color.b),
                    font: font,
                    maxWidth: imageWidth - (2 * textMargin),
                });
            }

            imagesOnCurrentPage++;
        }

        progressText.textContent = 'Generating PDF...';

        // Save PDF
        const pdfBytes = await pdfDoc.save();
        const pdfBlob = new Blob([pdfBytes], { type: 'application/pdf' });

        // Download
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `converted_images_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        resultDiv.innerHTML = `
            <div class="text-green-600">
                ✅ <strong>Images to PDF Conversion Successful!</strong><br>
                🖼️ Converted ${files.length} images to PDF<br>
                📍 Description Position: ${descriptionPosition}<br>
                📄 Images per Page: ${imagesPerPage}
            </div>
        `;

    } catch (error) {
        console.error('Multiple Images to PDF conversion failed:', error);
        resultDiv.innerHTML = `
            <div class="text-red-600">
                ❌ Conversion failed: ${error.message}
            </div>
        `;
    } finally {
        progressDiv.style.display = 'none';
        progressText.textContent = '';
    }
}

// Helper function to handle image previews and reordering
function initializeMultipleImagePreview() {
    const fileInput = document.getElementById('multipleImageToPdf-files');
    const filesCount = document.getElementById('multipleImageToPdf-files-count');
    const imagePreviews = document.getElementById('image-previews');
    const imageList = document.getElementById('image-list');
    
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        
        if (files.length === 0) {
            imagePreviews.classList.add('hidden');
            filesCount.textContent = 'No files selected';
            return;
        }
        
        filesCount.textContent = `${files.length} file(s) selected`;
        imageList.innerHTML = '';
        imagePreviews.classList.remove('hidden');
        
        files.forEach((file, index) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const imageDiv = document.createElement('div');
                imageDiv.className = 'image-preview border border-gray-200 rounded-lg p-3 bg-white';
                imageDiv.dataset.fileIndex = index;
                
                imageDiv.innerHTML = `
                    <div class="flex flex-col items-center">
                        <img src="${e.target.result}" alt="${file.name}" class="mb-2 max-h-32 object-contain border border-gray-300">
                        <div class="flex items-center justify-between w-full mt-2">
                            <span class="text-gray-600 text-xs truncate" title="${file.name}">${file.name}</span>
                            <div class="flex space-x-1">
                                <button type="button" class="move-up bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700">
                                    <i class="fas fa-arrow-up"></i>
                                </button>
                                <button type="button" class="move-down bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700">
                                    <i class="fas fa-arrow-down"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                `;
                
                imageList.appendChild(imageDiv);
                
                // Add event listeners for move buttons
                imageDiv.querySelector('.move-up').addEventListener('click', (event) => {
                    event.preventDefault();
                    const prev = imageDiv.previousSibling;
                    if (prev) {
                        imageList.insertBefore(imageDiv, prev);
                        updateImageOrder();
                    }
                });
                
                imageDiv.querySelector('.move-down').addEventListener('click', (event) => {
                    event.preventDefault();
                    const next = imageDiv.nextSibling;
                    if (next) {
                        imageList.insertBefore(next, imageDiv);
                        updateImageOrder();
                    }
                });
            };
            
            reader.readAsDataURL(file);
        });
        
        updateImageOrder();
    });
}

function updateImageOrder() {
    const imagePreviews = document.querySelectorAll('.image-preview');
    const imageOrder = Array.from(imagePreviews).map(preview => preview.dataset.fileIndex);
    document.getElementById('multipleImageToPdf-image-order').value = imageOrder.join(',');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeMultipleImagePreview();
    // Make sure this function exists or add it:
    if (typeof updateFileLabel === 'function') {
        updateFileLabel('multipleImageToPdf-files', 'multipleImageToPdf-files-count');
    }
});

