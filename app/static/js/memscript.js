    // Memory Monitoring Code
    document.addEventListener('DOMContentLoaded', function() {
        const ramBar = document.getElementById('ram-bar');
        const ramStats = document.getElementById('ram-stats');
        const romBar = document.getElementById('rom-bar');
        const romStats = document.getElementById('rom-stats');
    
        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
    
        function updateMemoryUI(data) {
            // Update RAM
            const ramPercent = data.ram.percent;
            if (ramBar) {
                ramBar.style.width = `${ramPercent}%`;
                ramBar.style.backgroundColor = ramPercent > 80 ? '#f44336' : '#4CAF50';
            }
            if (ramStats) {
                ramStats.innerHTML = `
                    <div>Used: ${formatBytes(data.ram.used)} / ${formatBytes(data.ram.total)}</div>
                    <div>Free: ${formatBytes(data.ram.free)}</div>
                    <div>Usage: ${ramPercent.toFixed(1)}%</div>
                `;
            }
    
            // Update ROM (Swap)
            const romPercent = data.rom.percent;
            if (romBar) {
                romBar.style.width = `${romPercent}%`;
                romBar.style.backgroundColor = romPercent > 80 ? '#f44336' : '#4CAF50';
            }
            if (romStats) {
                romStats.innerHTML = `
                    <div>Used: ${formatBytes(data.rom.used)} / ${formatBytes(data.rom.total)}</div>
                    <div>Free: ${formatBytes(data.rom.free)}</div>
                    <div>Usage: ${romPercent.toFixed(1)}%</div>
                `;
            }
        }
    
        // Connect to SSE endpoint  this commented out memory-usage  endpoint also commented out

        // const eventSource = new EventSource(`${BASE_URL}/memory-usage`);
        
        // eventSource.onmessage = (event) => {
        //     const data = JSON.parse(event.data);
        //     updateMemoryUI(data);
        // };
    
        // eventSource.onerror = () => {
        //     console.error('EventSource failed');
        //     // Attempt to reconnect after 2 seconds
        //     setTimeout(() => {
        //         eventSource.close();
        //         new EventSource('/memory-usage');
        //     }, 2000);
        // };
    });
    