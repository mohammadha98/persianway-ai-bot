/**
 * Main JavaScript file for Persian Agriculture Knowledge Base UI
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize any components that need JavaScript functionality
    initializeChatInterface();
    initializeSettingsPanel();
    initializeContributionForm();
});

/**
 * Chat Interface Functionality
 */
function initializeChatInterface() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // Only initialize if we're on the chat page
    if (!chatForm) return;

    // Handle chat form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (message) {
            // Add user message to chat
            addMessageToChat('user', message);
            chatInput.value = '';
            
            // Show loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message ai-message loading';
            loadingDiv.innerHTML = '<div class="spinner"></div>';
            chatMessages.appendChild(loadingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Send message to backend
            fetch('/ui/chat/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    user_id: document.getElementById('user-id')?.value || 'default-user'
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Remove loading indicator
                chatMessages.removeChild(loadingDiv);
                
                // Add AI response to chat
                if (data.answer) {
                    addMessageToChat('ai', data.answer);
                } else if (data.error) {
                    showToast('Error', data.error, 'error');
                }
            })
            .catch(error => {
                // Remove loading indicator
                chatMessages.removeChild(loadingDiv);
                console.error('Error:', error);
                showToast('Error', 'Failed to send message. Please try again.', 'error');
            });
        }
    });

    // Handle clear chat button
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', function() {
            chatMessages.innerHTML = '';
            showToast('Success', 'Chat cleared successfully', 'success');
        });
    }
}

/**
 * Add a message to the chat interface
 * @param {string} sender - 'user' or 'ai'
 * @param {string} message - The message text
 */
function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageWrapper = document.createElement('div');
    messageWrapper.className = `flex w-full mb-4 ${sender === 'user' ? 'justify-end' : 'justify-start'}`;

    const messageDiv = document.createElement('div');
    const messageClasses = 'max-w-lg p-3 rounded-lg shadow-md';
    if (sender === 'user') {
        messageDiv.className = `${messageClasses} bg-green-600 text-white`;
    } else {
        messageDiv.className = `${messageClasses} bg-gray-200 text-gray-800`;
    }

    // Format message with markdown if it's from AI
    if (sender === 'ai' && typeof marked !== 'undefined') {
        messageDiv.innerHTML = marked.parse(message);
    } else {
        const p = document.createElement('p');
        p.textContent = message;
        messageDiv.appendChild(p);
    }

    // Add timestamp
    const timeDiv = document.createElement('div');
    const timeClasses = 'text-xs mt-1';
    if (sender === 'user') {
        timeDiv.className = `${timeClasses} text-green-200 text-right`;
    } else {
        timeDiv.className = `${timeClasses} text-gray-400 text-left`;
    }
    timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.appendChild(timeDiv);

    messageWrapper.appendChild(messageDiv);
    chatMessages.appendChild(messageWrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Settings Panel Functionality
 */
function initializeSettingsPanel() {
    const processPdfForm = document.getElementById('process-pdf-form');
    const processExcelForm = document.getElementById('process-excel-form');
    const refreshStatusBtn = document.getElementById('refresh-status-btn');

    // Only initialize if we're on the settings page
    if (!processPdfForm && !processExcelForm) return;

    // Handle PDF processing form
    if (processPdfForm) {
        processPdfForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(processPdfForm);
            
            // Show loading state
            const submitBtn = processPdfForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            
            fetch('/ui/settings/process-pdf', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                // Reset button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
                
                if (data.success) {
                    processPdfForm.reset();
                    showToast('Success', 'PDF document submitted for processing', 'success');
                } else {
                    showToast('Error', data.error || 'Failed to process PDF', 'error');
                }
            })
            .catch(error => {
                // Reset button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
                
                console.error('Error:', error);
                showToast('Error', 'Failed to process PDF. Please try again.', 'error');
            });
        });
    }

    // Handle Excel processing form
    if (processExcelForm) {
        processExcelForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(processExcelForm);
            
            // Show loading state
            const submitBtn = processExcelForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            
            fetch('/ui/settings/process-excel', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                // Reset button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
                
                if (data.success) {
                    processExcelForm.reset();
                    showToast('Success', 'Excel file submitted for processing', 'success');
                } else {
                    showToast('Error', data.error || 'Failed to process Excel file', 'error');
                }
            })
            .catch(error => {
                // Reset button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
                
                console.error('Error:', error);
                showToast('Error', 'Failed to process Excel file. Please try again.', 'error');
            });
        });
    }

    // Handle refresh status button
    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', function() {
            // Show loading state
            refreshStatusBtn.disabled = true;
            refreshStatusBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
            
            fetch('/ui/settings/status')
            .then(response => response.json())
            .then(data => {
                // Reset button
                refreshStatusBtn.disabled = false;
                refreshStatusBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
                
                // Update status elements
                updateKnowledgeBaseStatus(data);
            })
            .catch(error => {
                // Reset button
                refreshStatusBtn.disabled = false;
                refreshStatusBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
                
                console.error('Error:', error);
                showToast('Error', 'Failed to refresh status. Please try again.', 'error');
            });
        });
    }
}

/**
 * Update knowledge base status elements with data from API
 * @param {Object} data - Status data from API
 */
function updateKnowledgeBaseStatus(data) {
    const statusBadge = document.getElementById('kb-status-badge');
    const totalDocsSpan = document.getElementById('total-docs');
    const pdfDocsSpan = document.getElementById('pdf-docs');
    const excelQASpan = document.getElementById('excel-qa');
    
    if (statusBadge) {
        // Update status badge
        statusBadge.className = 'badge';
        statusBadge.textContent = data.status;
        
        if (data.status === 'ready') {
            statusBadge.classList.add('bg-success');
        } else if (data.status === 'processing') {
            statusBadge.classList.add('bg-warning', 'text-dark');
        } else if (data.status === 'empty') {
            statusBadge.classList.add('bg-danger');
        }
    }
    
    // Update document counts
    if (totalDocsSpan) totalDocsSpan.textContent = data.document_counts.total || 0;
    if (pdfDocsSpan) pdfDocsSpan.textContent = data.document_counts.pdf || 0;
    if (excelQASpan) excelQASpan.textContent = data.document_counts.excel_qa || 0;
}

/**
 * Contribution Form Functionality
 */
function initializeContributionForm() {
    const contributionForm = document.getElementById('contribution-form');
    if (!contributionForm) return;

    const steps = Array.from(contributionForm.querySelectorAll('.form-step'));
    const nextButtons = Array.from(contributionForm.querySelectorAll('.next-step'));
    const prevButtons = Array.from(contributionForm.querySelectorAll('.prev-step'));
    const stepIndicators = Array.from(document.querySelectorAll('.step-indicator .step'));

    let currentStep = 1;

    const updateFormSteps = () => {
        steps.forEach(step => {
            step.classList.toggle('hidden', parseInt(step.dataset.step) !== currentStep);
            step.classList.toggle('active', parseInt(step.dataset.step) === currentStep);
        });
        updateStepIndicator();
    };

    const updateStepIndicator = () => {
        stepIndicators.forEach(indicator => {
            const step = parseInt(indicator.dataset.step);
            if (step === currentStep) {
                indicator.classList.add('bg-green-600', 'text-white');
                indicator.classList.remove('bg-gray-300', 'text-gray-600');
            } else if (step < currentStep) {
                indicator.classList.add('bg-green-600', 'text-white');
                indicator.classList.remove('bg-gray-300', 'text-gray-600');
            } else {
                indicator.classList.add('bg-gray-300', 'text-gray-600');
                indicator.classList.remove('bg-green-600', 'text-white');
            }
        });
    };

    nextButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Basic validation can be added here if needed
            if (currentStep < steps.length) {
                currentStep++;
                updateFormSteps();
            }
        });
    });

    prevButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (currentStep > 1) {
                currentStep--;
                updateFormSteps();
            }
        });
    });

    contributionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        // Handle form submission logic here
        // This is a placeholder. Full submission logic with fetch can be re-integrated.
        console.log('Form submitted');
        showToast('Success', 'Your contribution has been submitted for review.', 'success');
        contributionForm.reset();
        currentStep = 1;
        updateFormSteps();
    });

    updateFormSteps(); // Initial setup
}

/**
 * Show a toast notification
 * @param {string} title - The toast title
 * @param {string} message - The toast message
 * @param {string} type - The toast type ('success', 'error', 'warning', 'info')
 */
function showToast(title, message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastEl = document.createElement('div');
    toastEl.className = 'toast';
    toastEl.setAttribute('id', toastId);
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    // Set background color based on type
    let bgClass = 'bg-info';
    if (type === 'success') bgClass = 'bg-success';
    if (type === 'error') bgClass = 'bg-danger';
    if (type === 'warning') bgClass = 'bg-warning text-dark';
    
    // Create toast content
    toastEl.innerHTML = `
        <div class="toast-header ${bgClass} text-white">
            <strong class="me-auto">${title}</strong>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    // Add toast to container
    toastContainer.appendChild(toastEl);
    
    // Initialize and show toast
    const toast = new bootstrap.Toast(toastEl, {
        autohide: true,
        delay: 5000
    });
    toast.show();
    
    // Remove toast element after it's hidden
    toastEl.addEventListener('hidden.bs.toast', function() {
        toastEl.remove();
    });
}