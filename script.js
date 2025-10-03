let uploadedFile = null;
let allStudents = [];
let currentFilter = 'all';

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing...');
    
    // Upload handling
    const uploadButton = document.getElementById('upload-button');
    const analyzeButton = document.getElementById('analyze-button');
    const uploadStatus = document.getElementById('upload-status');

    if (uploadButton) {
        uploadButton.addEventListener('change', (e) => {
            console.log('File selected');
            if (e.target.files.length > 0) {
                uploadedFile = e.target.files[0];
                uploadStatus.textContent = `Selected file: ${uploadedFile.name}`;
                uploadStatus.style.display = 'block';
                analyzeButton.disabled = false;
            } else {
                uploadedFile = null;
                uploadStatus.style.display = 'none';
                analyzeButton.disabled = true;
            }
        });
    }

    if (analyzeButton) {
        analyzeButton.addEventListener('click', async () => {
            console.log('Analyze button clicked');
            await handleAnalyze();
        });
    }

    // Filter event listeners
    const filterAll = document.getElementById('filter-all');
    const filterLow = document.getElementById('filter-low');
    const filterMedium = document.getElementById('filter-medium');
    const filterHigh = document.getElementById('filter-high');

    if (filterAll) filterAll.addEventListener('click', () => filterStudents('all'));
    if (filterLow) filterLow.addEventListener('click', () => filterStudents('low'));
    if (filterMedium) filterMedium.addEventListener('click', () => filterStudents('medium'));
    if (filterHigh) filterHigh.addEventListener('click', () => filterStudents('high'));

    // Initialize dashboard
    fetchFeatures();
    fetchStudents();
});

async function handleAnalyze() {
    if (!uploadedFile) {
        alert('Please select a file to upload first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', uploadedFile);

    const uploadStatus = document.getElementById('upload-status');
    uploadStatus.textContent = 'Uploading and analyzing...';

    try {
        console.log('Starting upload...');
        const response = await fetch('http://localhost:5000/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('Upload response:', result);

        if (!response.ok) {
            alert(`Error: ${result.error || 'Unknown error'}`);
            uploadStatus.textContent = '';
            return;
        }

        uploadStatus.textContent = result.message;
        document.getElementById('analyze-button').disabled = true;
        document.getElementById('upload-button').value = '';
        uploadedFile = null;

        // Refresh data
        await fetchFeatures();
        await fetchStudents();

    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed: ' + error.message);
        uploadStatus.textContent = '';
    }
}

async function fetchFeatures() {
    try {
        console.log('Fetching features...');
        const response = await fetch('http://localhost:5000/api/features');
        if (!response.ok) throw new Error('Failed fetching features');
        const features = await response.json();
        renderFeatures(features);
        console.log('Features rendered');
    } catch (e) {
        console.error('Features error:', e);
        const featureList = document.getElementById('feature-list');
        if (featureList) {
            featureList.textContent = 'Error loading features';
        }
    }
}

function renderFeatures(features) {
    const container = document.getElementById('feature-list');
    if (!container) return;
    
    container.innerHTML = '';
    features.forEach(feature => {
        const item = document.createElement('div');
        item.className = 'feature-item';
        item.innerHTML = `
            <span>${feature.name}</span>
            <span>${feature.importance}%</span>
        `;
        container.appendChild(item);
    });
}

async function fetchStudents() {
    try {
        console.log('Fetching students...');
        const response = await fetch('http://localhost:5000/api/students');
        if (!response.ok) throw new Error('Cannot load students');
        const students = await response.json();
        console.log('Students fetched:', students.length);
        renderStudents(students);
        clearExplanation();
    } catch (e) {
        console.error('Students error:', e);
        const tbody = document.getElementById('student-tbody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="3">Error loading student data</td></tr>';
        }
    }
}

function renderStudents(students) {
    console.log('Rendering students:', students.length);
    allStudents = students;
    displayStudents(students);
    updateFilterStats();
}

function displayStudents(students) {
    const tbody = document.getElementById('student-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    students.forEach(student => {
        const row = document.createElement('tr');
        row.dataset.studentId = student.id;
        row.className = 'student-row';

        row.innerHTML = `
            <td>${student.name} (${student.id})</td>
            <td><span class="risk-${student.riskClass}">${student.dropout_risk}%</span></td>
            <td><button class="view-details-btn" data-student-id="${student.id}">View Details</button></td>
        `;

        // Add event listeners for both row click and button click
        row.addEventListener('click', (e) => {
            // Don't trigger if button was clicked
            if (e.target.tagName !== 'BUTTON') {
                selectStudent(student.id);
            }
        });

        // Add specific button click handler
        const button = row.querySelector('.view-details-btn');
        if (button) {
            button.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent row click
                console.log('View details button clicked for student:', student.id);
                selectStudent(student.id);
            });
        }

        tbody.appendChild(row);
    });

    console.log('Students displayed successfully');
}

function updateFilterStats() {
    const total = allStudents.length;
    const low = allStudents.filter(s => s.riskClass === 'low').length;
    const medium = allStudents.filter(s => s.riskClass === 'medium').length;
    const high = allStudents.filter(s => s.riskClass === 'high').length;
    
    const filterStats = document.getElementById('filter-stats');
    if (filterStats) {
        filterStats.innerHTML = `Total: ${total} | Low Risk: ${low} | Moderate: ${medium} | High Risk: ${high}`;
    }
}

function filterStudents(filterType) {
    console.log('Filtering students:', filterType);
    currentFilter = filterType;
    
    // Update active button
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`filter-${filterType}`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // Filter students
    let filteredStudents = allStudents;
    if (filterType !== 'all') {
        filteredStudents = allStudents.filter(student => student.riskClass === filterType);
    }
    
    displayStudents(filteredStudents);
    
    // Update stats
    const filterStats = document.getElementById('filter-stats');
    if (filterStats) {
        filterStats.innerHTML = `Showing ${filteredStudents.length} of ${allStudents.length} students`;
    }
}

// Student explanation - FIXED VERSION
async function selectStudent(studentId) {
    console.log(`Selecting student ${studentId} for explanation`);
    
    const explanationContent = document.getElementById('explanation-content');
    if (!explanationContent) {
        console.error('Explanation content element not found');
        return;
    }

    // Show loading
    explanationContent.innerHTML = '<p>Loading explanation...</p>';
    
    try {
        console.log('Fetching explanation...');
        const response = await fetch(`http://localhost:5000/api/explain_student?student_id=${studentId}`);
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const explanation = await response.json();
        console.log('Explanation received:', explanation.student_id, explanation.predicted_dropout_risk_percent);

        if (explanation.error) {
            explanationContent.innerHTML = `<p style="color: red;">Error: ${explanation.error}</p>`;
            return;
        }

        // Display the explanation
        let html = `
            <h3>👩‍🎓 ${explanation.counselor_name || 'Dr. Sarah Chen'}</h3>
            <p><strong>Student ${explanation.student_id + 1} - Dropout Risk: ${explanation.predicted_dropout_risk_percent}%</strong></p>
            
            <h4>📋 Institutional Assessment</h4>
            <div style="background: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd; margin: 10px 0;">
                <pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">${explanation.ai_explanation}</pre>
            </div>
            
            <details style="margin-top: 15px;">
                <summary><strong>🔍 Technical Analysis Details</strong></summary>
                <ul style="margin-top: 10px; padding-left: 20px;">`;
        
        if (explanation.technical_factors && explanation.technical_factors.length > 0) {
            explanation.technical_factors.forEach(feat => {
                html += `<li style="margin-bottom: 5px;"><b>${feat.feature}</b> = ${feat.feature_value} : ${feat.interpretation} (SHAP=${feat.shap_value.toFixed(3)})</li>`;
            });
        } else {
            html += '<li>No technical factors available</li>';
        }
        
        html += `</ul></details>`;

        explanationContent.innerHTML = html;

        // Highlight selected row
        document.querySelectorAll('.student-row').forEach(r => r.classList.remove('selected'));
        const selectedRow = document.querySelector(`[data-student-id="${studentId}"]`);
        if (selectedRow) {
            selectedRow.classList.add('selected');
        }

        // Scroll to explanation
        const explanationSection = document.getElementById('explanation-section');
        if (explanationSection) {
            explanationSection.scrollIntoView({behavior: 'smooth'});
        }

        console.log('Explanation displayed successfully');

    } catch (e) {
        console.error('Student explanation error:', e);
        explanationContent.innerHTML = `<p style="color: red;">Error loading explanation: ${e.message}</p>`;
    }
}

function clearExplanation() {
    const explanationContent = document.getElementById('explanation-content');
    if (explanationContent) {
        explanationContent.innerHTML = '<p>Select a student from the table to view detailed assessment.</p>';
    }
}

// Global function for onclick handlers in HTML
window.selectStudent = selectStudent;
