// PDF Library Manager - Uses ALL 8 libraries
class PDFLibraryManager {
    constructor() {
        this.libraries = {
            pdfjs: { loaded: false, loading: false, lib: null },
            // pdfjsWorker: { loaded: false, loading: false },
            pdfLib: { loaded: false, loading: false, lib: null },
            pptxgen: { loaded: false, loading: false, lib: null },
            jsPDF: { loaded: false, loading: false, lib: null },
            jszip: { loaded: false, loading: false, lib: null },
            fileSaver: { loaded: false, loading: false, lib: null },
            downloadjs: { loaded: false, loading: false, lib: null }
        };
        
        // Set PDF.js worker path
        // if (typeof pdfjsLib !== 'undefined') {
        //     pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/js/pdflibra/pdf.worker.min.js';
        //     console.log("pdf worker loaded for pdfjs globally");
            
        // }
    }

    // Load individual library
    async loadLibrary(name) {
        if (this.libraries[name].loaded) return this.libraries[name].lib;
        if (this.libraries[name].loading) {
            return this.waitForLibrary(name);
        }

        this.libraries[name].loading = true;
        console.log(`Loading ${name} library...`);

        try {
            switch (name) {
                case 'pdfjs':
                    await this.loadPDFJS();
                    break;
                // case 'pdfjsWorker':
                //     await this.loadPDFJSWorker();
                //     break;
                case 'pdfLib':
                    await this.loadPDFLib();
                    break;
                case 'pptxgen':
                    await this.loadPptxGen();
                    break;
                case 'jsPDF':
                    await this.loadJsPDF();
                    break;
                case 'jszip':
                    await this.loadJSZip();
                    break;
                case 'fileSaver':
                    await this.loadFileSaver();
                    break;
                case 'downloadjs':
                    await this.loadDownloadJS();
                    break;
                default:
                    throw new Error(`Unknown library: ${name}`);
            }
            
            return this.libraries[name].lib;
        } catch (error) {
            this.libraries[name].loading = false;
            throw error;
        }
    }

    // Load multiple libraries at once
    async loadLibraries(libraryNames) {
        const promises = libraryNames.map(name => this.loadLibrary(name));
        return Promise.all(promises);
    }

    // Preload libraries for specific tools
    preloadForTool(toolName) {
        const toolDependencies = {
            // Chat tools
            'chat-section': [], // No PDF libraries needed for chat
            
            // PDF manipulation tools
            'merge-section': ['pdfLib', 'pdfjs', 'jszip', 'fileSaver'],
            'compress-section': ['pdfjs', 'pdfLib', 'fileSaver'],
            'encrypt-section': ['pdfLib', 'fileSaver'],
            'split-section': ['pdfLib', 'jszip', 'fileSaver'],
            'deletePages-section': ['pdfjs', 'pdfLib', 'fileSaver'],
            'reorder-section': ['pdfjs', 'pdfLib', 'fileSaver'],
            'pageNumbers-section': ['pdfLib', 'fileSaver'],
            'signature-section': ['pdfjs', 'pdfLib', 'fileSaver'],
            'removePassword-section': ['pdfLib', 'fileSaver'],
            
            // Conversion tools
            'pdfToImages-section': ['pdfjs', 'jszip', 'fileSaver'],
            'pdfToWord-section': ['pdfjs', 'fileSaver'],
            'pdfToExcel-section': ['pdfjs', 'fileSaver'],
            'pdfToPpt-section': ['pdfjs', 'pptxgen', 'fileSaver'],
            'imageToPdf-section': ['jsPDF', 'pdfLib', 'fileSaver'],
            'removeBackground-section': [] // Image processing, no PDF libs
        };

        const libs = toolDependencies[toolName] || [];
        console.log(`Preloading libraries for ${toolName}:`, libs);
        
        // Load in background
        libs.forEach(lib => this.loadLibrary(lib).catch(console.warn));
    }

    // Individual library loaders
    async loadPDFJS() {
        return new Promise((resolve, reject) => {
            if (typeof pdfjsLib !== 'undefined') {
                // Set worker path for PDF.js
                pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/js/pdflibra/pdf.worker.min.js';
                this.libraries.pdfjs.loaded = true;
                this.libraries.pdfjs.lib = pdfjsLib;
                resolve(pdfjsLib);
                return;
            }
    
            const script = document.createElement('script');
            script.src = '/static/js/pdflibra/pdf.min.js';
            script.onload = () => {
                // Set worker path AFTER PDF.js loads
                window.pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/js/pdflibra/pdf.worker.min.js';
                this.libraries.pdfjs.loaded = true;
                this.libraries.pdfjs.lib = window.pdfjsLib;
                resolve(window.pdfjsLib);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // async loadPDFJSWorker() {
    //     return new Promise((resolve, reject) => {
    //         const script = document.createElement('script');
    //         script.src = '/static/js/pdflibra/pdf.worker.min.js';
    //         script.onload = () => {
    //             this.libraries.pdfjsWorker.loaded = true;
    //             resolve();
    //         };
    //         script.onerror = reject;
    //         document.head.appendChild(script);
    //     });
    // }

    async loadPDFLib() {
        return new Promise((resolve, reject) => {
            if (typeof PDFLib !== 'undefined') {
                this.libraries.pdfLib.loaded = true;
                this.libraries.pdfLib.lib = PDFLib;
                resolve(PDFLib);
                return;
            }

            const script = document.createElement('script');
            script.src = '/static/js/pdflibra/pdf-lib.min.js';
            script.onload = () => {
                this.libraries.pdfLib.loaded = true;
                this.libraries.pdfLib.lib = window.PDFLib;
                resolve(window.PDFLib);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async loadPptxGen() {
        return new Promise((resolve, reject) => {
            if (typeof PptxGenJS !== 'undefined') {
                this.libraries.pptxgen.loaded = true;
                this.libraries.pptxgen.lib = PptxGenJS;
                resolve(PptxGenJS);
                return;
            }

            const script = document.createElement('script');
            script.src = '/static/js/pdflibra/pptxgen.bundle.js';
            script.onload = () => {
                this.libraries.pptxgen.loaded = true;
                this.libraries.pptxgen.lib = window.PptxGenJS;
                resolve(window.PptxGenJS);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async loadJsPDF() {
        return new Promise((resolve, reject) => {
            if (typeof jspdf !== 'undefined') {
                this.libraries.jsPDF.loaded = true;
                this.libraries.jsPDF.lib = jspdf.jsPDF;
                resolve(jspdf.jsPDF);
                return;
            }

            const script = document.createElement('script');
            script.src = '/static/js/pdflibra/jspdf.umd.min.js';
            script.onload = () => {
                this.libraries.jsPDF.loaded = true;
                this.libraries.jsPDF.lib = window.jspdf.jsPDF;
                resolve(window.jspdf.jsPDF);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async loadJSZip() {
        return new Promise((resolve, reject) => {
            if (typeof JSZip !== 'undefined') {
                this.libraries.jszip.loaded = true;
                this.libraries.jszip.lib = JSZip;
                resolve(JSZip);
                return;
            }

            const script = document.createElement('script');
            script.src = '/static/js/pdflibra/jszip.min.js';
            script.onload = () => {
                this.libraries.jszip.loaded = true;
                this.libraries.jszip.lib = window.JSZip;
                resolve(window.JSZip);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async loadFileSaver() {
        return new Promise((resolve, reject) => {
            if (typeof saveAs !== 'undefined') {
                this.libraries.fileSaver.loaded = true;
                this.libraries.fileSaver.lib = saveAs;
                resolve(saveAs);
                return;
            }

            const script = document.createElement('script');
            script.src = '/static/js/pdflibra/FileSaver.min.js';
            script.onload = () => {
                this.libraries.fileSaver.loaded = true;
                this.libraries.fileSaver.lib = window.saveAs;
                resolve(window.saveAs);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async loadDownloadJS() {
        return new Promise((resolve, reject) => {
            if (typeof download !== 'undefined') {
                this.libraries.downloadjs.loaded = true;
                this.libraries.downloadjs.lib = download;
                resolve(download);
                return;
            }

            const script = document.createElement('script');
            script.src = '/static/js/pdflibra/download.js';
            script.onload = () => {
                this.libraries.downloadjs.loaded = true;
                this.libraries.downloadjs.lib = window.download;
                resolve(window.download);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // Wait for library to load
    async waitForLibrary(name) {
        return new Promise((resolve) => {
            const check = () => {
                if (this.libraries[name].loaded) {
                    resolve(this.libraries[name].lib);
                } else {
                    setTimeout(check, 50);
                }
            };
            check();
        });
    }

    // Get library status
    getStatus() {
        const status = {};
        Object.keys(this.libraries).forEach(lib => {
            status[lib] = {
                loaded: this.libraries[lib].loaded,
                loading: this.libraries[lib].loading
            };
        });
        return status;
    }
}

// Global instance
window.pdfLibraryManager = new PDFLibraryManager();