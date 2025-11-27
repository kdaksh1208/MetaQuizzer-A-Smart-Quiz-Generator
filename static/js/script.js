// Global variables
let currentSessionId = null;
let selectedAnswer = null;

// DOM Elements
const generateBtn = document.getElementById('generateBtn');
const topicInput = document.getElementById('topicInput');
const contentInput = document.getElementById('contentInput');
const quizInputSection = document.getElementById('quizInputSection');
const quizActiveSection = document.getElementById('quizActiveSection');
const resultsSection = document.getElementById('resultsSection');
const submitAnswerBtn = document.getElementById('submitAnswerBtn');
const nextQuestionBtn = document.getElementById('nextQuestionBtn');
const newQuizBtn = document.getElementById('newQuizBtn');

// Stats elements
const totalScoreEl = document.getElementById('totalScore');
const averageScoreEl = document.getElementById('averageScore');
const flashcardSetsEl = document.getElementById('flashcardSets');
const studyTimeEl = document.getElementById('studyTime');

// Quiz elements
const currentQuestionNumEl = document.getElementById('currentQuestionNum');
const difficultyBadgeEl = document.getElementById('difficultyBadge');
const progressFillEl = document.getElementById('progressFill');
const questionTextEl = document.getElementById('questionText');
const optionsContainerEl = document.getElementById('optionsContainer');
const feedbackSection = document.getElementById('feedbackSection');
const feedbackText = document.getElementById('feedbackText');

// Start Quiz
generateBtn.addEventListener('click', async () => {
    const topic = topicInput.value.trim();
    const content = contentInput.value.trim();
    
    if (!topic || !content) {
        alert('Please enter both topic and content!');
        return;
    }
    
    generateBtn.disabled = true;
    generateBtn.innerHTML = '⏳ Starting Quiz...';
    
    try {
        const response = await fetch('/api/start-quiz', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic, content })
        });
        
        const data = await response.json();
        
        if (data.session_id) {
            currentSessionId = data.session_id;
            quizInputSection.style.display = 'none';
            quizActiveSection.style.display = 'block';
            loadNextQuestion();
        } else {
            alert('Error starting quiz: ' + (data.error || 'Unknown error'));
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<span>🤖</span> Generate Quiz with AI';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error starting quiz. Please try again.');
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<span>🤖</span> Generate Quiz with AI';
    }
});

// Load Next Question
async function loadNextQuestion() {
    selectedAnswer = null;
    submitAnswerBtn.disabled = true;
    submitAnswerBtn.textContent = 'Submit Answer';
    feedbackSection.style.display = 'none';
    questionTextEl.textContent = 'Loading question...';
    optionsContainerEl.innerHTML = '<p style="text-align: center; color: #666;">🤖 Generating your next question...</p>';
    
    try {
        const response = await fetch('/api/get-question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });
        
        const data = await response.json();
        
        console.log('Question data received:', data); // Debug log
        
        if (data.completed) {
            showResults();
            return;
        }
        
        // Update UI
        currentQuestionNumEl.textContent = data.question_number;
        questionTextEl.textContent = data.question;
        difficultyBadgeEl.textContent = data.difficulty.charAt(0).toUpperCase() + data.difficulty.slice(1);
        difficultyBadgeEl.className = `difficulty-badge ${data.difficulty}`;
        
        // Update progress
        const progress = (data.question_number / data.total_questions) * 100;
        progressFillEl.style.width = `${progress}%`;
        
        // Display options
        displayOptions(data.options);
        
    } catch (error) {
        console.error('Error loading question:', error);
        questionTextEl.textContent = 'Error loading question. Please refresh the page and try again.';
        optionsContainerEl.innerHTML = '<p style="color: red;">Unable to load question. Please check your connection.</p>';
    }
}

// Display Options
function displayOptions(options) {
    console.log('Displaying options:', options); // Debug log
    
    optionsContainerEl.innerHTML = '';
    
    if (!options || typeof options !== 'object') {
        console.error('Invalid options format:', options);
        optionsContainerEl.innerHTML = '<p style="color: red;">Error: Invalid options format</p>';
        return;
    }
    
    // Sort keys to ensure consistent order (A, B, C, D)
    const sortedKeys = Object.keys(options).sort();
    
    sortedKeys.forEach(key => {
        const optionText = options[key];
        
        // Debug log
        console.log(`Option ${key}:`, optionText);
        
        const optionBtn = document.createElement('button');
        optionBtn.className = 'option-btn';
        optionBtn.setAttribute('data-answer', key);
        
        optionBtn.innerHTML = `
            <span class="option-letter">${key}</span>
            <span class="option-text">${optionText}</span>
        `;
        
        optionBtn.addEventListener('click', () => selectOption(key, optionBtn));
        optionsContainerEl.appendChild(optionBtn);
    });
}

// Select Option
function selectOption(answer, buttonElement) {
    // Remove previous selection
    document.querySelectorAll('.option-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    // Add selection to clicked button
    buttonElement.classList.add('selected');
    selectedAnswer = answer;
    submitAnswerBtn.disabled = false;
}

// Submit Answer
submitAnswerBtn.addEventListener('click', async () => {
    if (!selectedAnswer) return;
    
    submitAnswerBtn.disabled = true;
    submitAnswerBtn.textContent = 'Submitting...';
    
    try {
        const response = await fetch('/api/submit-answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                answer: selectedAnswer,
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        
        console.log('Submit answer response:', data); // Debug log
        
        // Show feedback
        showFeedback(data.is_correct, data.correct_answer);
        
        // Update stats
        updateStats(data.metrics);
        
        // Disable all option buttons and show correct/incorrect
        document.querySelectorAll('.option-btn').forEach(btn => {
            btn.disabled = true;
            const btnAnswer = btn.getAttribute('data-answer');
            
            if (btnAnswer === data.correct_answer) {
                btn.classList.add('correct');
            } else if (btnAnswer === selectedAnswer && !data.is_correct) {
                btn.classList.add('incorrect');
            }
        });
        
    } catch (error) {
        console.error('Error submitting answer:', error);
        alert('Error submitting answer. Please try again.');
        submitAnswerBtn.disabled = false;
        submitAnswerBtn.textContent = 'Submit Answer';
    }
});

// Show Feedback
function showFeedback(isCorrect, correctAnswer) {
    feedbackSection.style.display = 'block';
    feedbackSection.className = isCorrect ? 'feedback correct' : 'feedback incorrect';
    
    if (isCorrect) {
        feedbackText.innerHTML = '✅ <strong>Correct!</strong> Great job! Keep it up!';
    } else {
        feedbackText.innerHTML = `❌ <strong>Incorrect.</strong> The correct answer was <strong>${correctAnswer}</strong>. Review the material and try again!`;
    }
}

// Next Question
nextQuestionBtn.addEventListener('click', () => {
    loadNextQuestion();
});

// Update Stats
function updateStats(metrics) {
    if (metrics) {
        totalScoreEl.textContent = metrics.total_score || 0;
        averageScoreEl.textContent = `${Math.round(metrics.average_score) || 0}%`;
        studyTimeEl.textContent = `${metrics.total_time || 0}m`;
        flashcardSetsEl.textContent = '1';
    }
}

// Show Results
function showResults() {
    quizActiveSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Get final stats
    fetch('/api/get-metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('finalScore').textContent = data.total_score;
        document.getElementById('finalAverage').textContent = `${Math.round(data.average_score)}%`;
        document.getElementById('finalTime').textContent = `${data.study_time}m`;
    })
    .catch(error => {
        console.error('Error fetching final metrics:', error);
    });
}

// New Quiz
newQuizBtn.addEventListener('click', () => {
    location.reload();
});
