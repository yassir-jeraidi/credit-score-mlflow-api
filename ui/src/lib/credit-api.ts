/**
 * Credit Score API Service
 * 
 * Handles communication with the FastAPI backend for credit predictions
 */

import type {
    CreditApplication,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class CreditApiService {
    private baseUrl: string;
    private token: string | null = null;

    constructor(baseUrl: string = API_BASE_URL) {
        this.baseUrl = baseUrl;
    }

    /**
     * Set authentication token for API requests
     */
    setToken(token: string) {
        this.token = token;
    }

    /**
     * Get authorization headers
     */
    private getHeaders(): HeadersInit {
        const headers: HeadersInit = {
            'Content-Type': 'application/json',
        };
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        return headers;
    }

    /**
     * Check API health status
     */
    async checkHealth(): Promise<HealthResponse> {
        const response = await fetch(`${this.baseUrl}/api/v1/health`, {
            method: 'GET',
            headers: this.getHeaders(),
        });

        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }

        return response.json();
    }

    /**
     * Get model information
     */
    async getModelInfo(): Promise<ModelInfoResponse> {
        const response = await fetch(`${this.baseUrl}/api/v1/model/info`, {
            method: 'GET',
            headers: this.getHeaders(),
        });

        if (!response.ok) {
            throw new Error(`Failed to get model info: ${response.statusText}`);
        }

        return response.json();
    }

    /**
     * Make a credit prediction
     */
    async predict(application: CreditApplication): Promise<PredictionResponse> {
        const response = await fetch(`${this.baseUrl}/api/v1/predict`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify(application),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Prediction failed');
        }

        return response.json();
    }

    /**
     * Make batch predictions
     */
    async predictBatch(applications: CreditApplication[]): Promise<{
        predictions: PredictionResponse[];
        total_processed: number;
        processing_time_ms: number;
    }> {
        const response = await fetch(`${this.baseUrl}/api/v1/predict/batch`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({ applications }),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Batch prediction failed');
        }

        return response.json();
    }

    /**
     * Login to get authentication token
     */
    async login(email: string, password: string): Promise<string> {
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);

        const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Login failed');
        }

        const data = await response.json();
        this.token = data.access_token;
        return data.access_token;
    }
}

export const creditApi = new CreditApiService();
export default CreditApiService;
