document.addEventListener('DOMContentLoaded', function() {
    const refreshBtn = document.getElementById('refresh-status-btn');
    const pdfForm = document.getElementById('process-pdf-form');
    const excelForm = document.getElementById('process-excel-form');

    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            // Add a spinning animation to the refresh icon
            const icon = refreshBtn.querySelector('i');
            icon.classList.add('fa-spin');

            fetch('/api/kb/status')
                .then(response => response.json())
                .then(data => {
                    updateStatus(data);
                })
                .catch(error => {
                    console.error('Error refreshing status:', error);
                    alert('Failed to refresh status. Please check the console for details.');
                })
                .finally(() => {
                    // Remove the spinning animation
                    icon.classList.remove('fa-spin');
                });
        });
    }

    if (pdfForm) {
        pdfForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleFileUpload(this, '/api/kb/process-pdf');
        });
    }

    if (excelForm) {
        excelForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleFileUpload(this, '/api/kb/process-excel');
        });
    }

    function handleFileUpload(form, url) {
        const formData = new FormData(form);
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;

        // Disable button and show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = `<i class="fas fa-spinner fa-spin mr-2"></i> Processing...`;

        fetch(url, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.detail || 'File processing failed') });
            }
            return response.json();
        })
        .then(data => {
            alert(data.message || 'File processed successfully!');
            form.reset();
            // Optionally, refresh status after upload
            refreshBtn.click(); 
        })
        .catch(error => {
            console.error('Error uploading file:', error);
            alert(`Error: ${error.message}`);
        })
        .finally(() => {
            // Restore button state
            submitButton.disabled = false;
            submitButton.innerHTML = originalButtonText;
        });
    }

    function updateStatus(data) {
        // Update status indicator and badge
        const statusIndicator = document.getElementById('status-indicator');
        const statusBadge = document.getElementById('kb-status-badge');
        
        statusIndicator.className = `status-indicator status-${data.status}`;
        statusBadge.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
        
        statusBadge.className = `px-3 py-1 text-sm font-medium rounded-full `;
        if (data.status === 'ready') {
            statusBadge.classList.add('bg-green-100', 'text-green-800');
        } else if (data.status === 'processing') {
            statusBadge.classList.add('bg-yellow-100', 'text-yellow-800');
        } else if (data.status === 'empty') {
            statusBadge.classList.add('bg-red-100', 'text-red-800');
        }

        // Update document counts
        document.getElementById('total-docs').textContent = data.document_counts.total;
        document.getElementById('pdf-docs').textContent = data.document_counts.pdf;
        document.getElementById('excel-qa').textContent = data.document_counts.excel_qa;
    }
});