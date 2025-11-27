// Login Form Handler
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const remember = document.getElementById('remember').checked;
        const loginBtn = document.getElementById('loginBtn');
        const errorMessage = document.getElementById('errorMessage');
        
        loginBtn.disabled = true;
        loginBtn.textContent = 'Logging in...';
        errorMessage.style.display = 'none';
        
        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, password, remember })
            });
            
            const data = await response.json();
            
            if (data.success) {
                window.location.href = '/';
            } else {
                errorMessage.textContent = data.message;
                errorMessage.style.display = 'block';
                loginBtn.disabled = false;
                loginBtn.textContent = 'Login';
            }
        } catch (error) {
            errorMessage.textContent = 'An error occurred. Please try again.';
            errorMessage.style.display = 'block';
            loginBtn.disabled = false;
            loginBtn.textContent = 'Login';
        }
    });
}

// Signup Form Handler
const signupForm = document.getElementById('signupForm');
if (signupForm) {
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const signupBtn = document.getElementById('signupBtn');
        const errorMessage = document.getElementById('errorMessage');
        
        errorMessage.style.display = 'none';
        
        // Validate passwords match
        if (password !== confirmPassword) {
            errorMessage.textContent = 'Passwords do not match!';
            errorMessage.style.display = 'block';
            return;
        }
        
        // Validate password length
        if (password.length < 6) {
            errorMessage.textContent = 'Password must be at least 6 characters!';
            errorMessage.style.display = 'block';
            return;
        }
        
        signupBtn.disabled = true;
        signupBtn.textContent = 'Creating Account...';
        
        try {
            const response = await fetch('/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, email, password })
            });
            
            const data = await response.json();
            
            if (data.success) {
                window.location.href = '/';
            } else {
                errorMessage.textContent = data.message;
                errorMessage.style.display = 'block';
                signupBtn.disabled = false;
                signupBtn.textContent = 'Create Account';
            }
        } catch (error) {
            errorMessage.textContent = 'An error occurred. Please try again.';
            errorMessage.style.display = 'block';
            signupBtn.disabled = false;
            signupBtn.textContent = 'Create Account';
        }
    });
}
