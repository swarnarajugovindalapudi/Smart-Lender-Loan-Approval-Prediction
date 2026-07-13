document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('loanForm');
    
    if (form) {
        form.addEventListener('submit', (e) => {
            let isValid = true;
            
            // Clear all previous errors
            document.querySelectorAll('.error-message').forEach(el => el.textContent = '');
            
            // 1. Validate Dropdowns
            const selects = form.querySelectorAll('select');
            selects.forEach(select => {
                if (!select.value) {
                    showError(select.id, 'Please select an option.');
                    isValid = false;
                }
            });

            // 2. Validate Numerical Inputs
            const appIncome = document.getElementById('ApplicantIncome');
            if (appIncome && Number(appIncome.value) < 0) {
                showError('ApplicantIncome', 'Income cannot be negative.');
                isValid = false;
            }

            const coAppIncome = document.getElementById('CoapplicantIncome');
            if (coAppIncome && Number(coAppIncome.value) < 0) {
                showError('CoapplicantIncome', 'Income cannot be negative.');
                isValid = false;
            }

            const loanAmount = document.getElementById('LoanAmount');
            if (loanAmount && Number(loanAmount.value) <= 0) {
                showError('LoanAmount', 'Loan amount must be greater than 0.');
                isValid = false;
            }

            const term = document.getElementById('Loan_Amount_Term');
            if (term && Number(term.value) <= 0) {
                showError('Loan_Amount_Term', 'Term must be greater than 0.');
                isValid = false;
            }

            if (!isValid) {
                e.preventDefault(); // Prevent form submission
            }
        });
    }
});

function showError(inputId, message) {
    const errorSpan = document.getElementById(`error-${inputId}`);
    if (errorSpan) {
        errorSpan.textContent = message;
    }
}
