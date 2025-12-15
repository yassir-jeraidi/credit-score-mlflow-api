/**
 * Type definitions for Credit Score API integration
 */

export interface CreditApplication {
    age: number;
    income: number;
    employment_length: number;
    loan_amount: number;
    loan_intent: LoanIntent;
    home_ownership: HomeOwnership;
    credit_history_length: number;
    num_credit_lines: number;
    derogatory_marks: number;
    total_debt: number;
}

export type LoanIntent =
    | 'PERSONAL'
    | 'EDUCATION'
    | 'MEDICAL'
    | 'VENTURE'
    | 'HOMEIMPROVEMENT'
    | 'DEBTCONSOLIDATION';

export type HomeOwnership = 'RENT' | 'OWN' | 'MORTGAGE' | 'OTHER';

export interface PredictionResponse {
    application_id: string;
    prediction: 'APPROVED' | 'REJECTED';
    confidence: number;
    risk_score: number;
    approval_probability: number;
    rejection_probability: number;
    model_version: string;
    timestamp: string;
}

export interface HealthResponse {
    status: string;
    version: string;
    timestamp: string;
}

export interface ModelInfoResponse {
    model_name: string;
    model_stage: string;
    model_version: string | null;
    is_loaded: boolean;
    features: string[];
}

export interface ApiError {
    detail: string;
}
