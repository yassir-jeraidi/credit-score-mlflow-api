import {
  streamText,
  type UIMessage,
  convertToModelMessages,
  stepCountIs,
} from "ai";
import { google } from "@ai-sdk/google";
import { z } from "zod";
import { cookies } from "next/headers";
import { decrypt } from "@/lib/auth";
import type { CreditApplication, PredictionResponse } from "@/lib/types";

// Allow streaming responses up to 60 seconds
export const maxDuration = 60;

const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000";

/**
 * Make a prediction call to the FastAPI backend with authentication
 */
async function makeCreditPrediction(
  application: CreditApplication,
  accessToken: string,
): Promise<PredictionResponse> {
  if (!accessToken) {
    throw new Error("Authentication required. Please log in.");
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify(application),
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Prediction failed");
  }

  return response.json();
}

/**
 * Credit application schema for validation
 */
const creditApplicationSchema = z.object({
  age: z
    .number()
    .min(18)
    .max(100)
    .describe("Age of the applicant (18-100 years)"),
  income: z.number().positive().describe("Annual income in currency units"),
  employment_length: z.number().min(0).max(50).describe("Years of employment"),
  loan_amount: z.number().positive().describe("Requested loan amount"),
  loan_intent: z
    .enum([
      "PERSONAL",
      "EDUCATION",
      "MEDICAL",
      "VENTURE",
      "HOMEIMPROVEMENT",
      "DEBTCONSOLIDATION",
    ])
    .describe("Purpose of the loan"),
  home_ownership: z
    .enum(["RENT", "OWN", "MORTGAGE", "OTHER"])
    .describe("Home ownership status"),
  credit_history_length: z
    .number()
    .min(0)
    .max(50)
    .describe("Years of credit history"),
  num_credit_lines: z
    .number()
    .min(0)
    .max(50)
    .describe("Number of credit lines/accounts"),
  derogatory_marks: z
    .number()
    .min(0)
    .max(20)
    .describe("Number of derogatory marks on credit report"),
  total_debt: z.number().min(0).describe("Total existing debt"),
});

export async function POST(req: Request) {
  const {
    messages,
    model,
  }: {
    messages: UIMessage[];
    model?: string;
  } = await req.json();

  // Get access token from session cookie
  const cookieStore = await cookies();
  const sessionCookie = cookieStore.get("session")?.value;
  const session = await decrypt(sessionCookie);
  const accessToken = session?.accessToken;

  const result = streamText({
    model: google("gemini-2.5-flash"),
    messages: convertToModelMessages(messages),
    system: `You are a helpful credit scoring assistant for a financial services application. 
Your role is to:
1. Help users understand credit scoring and loan applications
2. Collect credit application information from users through conversation
3. Use the analyze_credit_application tool to make predictions when you have all required information

When collecting information, ask for these details naturally:
- Age (18-100 years)
- Annual income
- Years of employment
- Requested loan amount
- Loan purpose (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
- Home ownership status (RENT, OWN, MORTGAGE, OTHER)
- Years of credit history
- Number of credit lines/accounts
- Number of derogatory marks (negative items on credit report)
- Total existing debt

Once you have all the information, use the analyze_credit_application tool to get a prediction.
After receiving a prediction, explain the results clearly to the user, including:
- The decision (APPROVED/REJECTED)
- Confidence level
- Risk assessment
- Suggestions for improvement if rejected

Be professional, helpful, and provide financial guidance while being clear that this is a demonstration system.`,
    tools: {
      analyze_credit_application: {
        description:
          "Analyze a credit application and get a prediction from the ML model",
        inputSchema: creditApplicationSchema,
        execute: async (
          application: z.infer<typeof creditApplicationSchema>,
        ) => {
          if (!accessToken) {
            return {
              success: false,
              error:
                "Authentication required. Please log in to analyze credit applications.",
            };
          }
          try {
            const prediction = await makeCreditPrediction(
              application as CreditApplication,
              accessToken,
            );
            return {
              success: true,
              prediction: prediction.prediction,
              confidence: Math.round(prediction.confidence * 100),
              risk_score: Math.round(prediction.risk_score * 100),
              approval_probability: Math.round(
                prediction.approval_probability * 100,
              ),
              rejection_probability: Math.round(
                prediction.rejection_probability * 100,
              ),
              model_version: prediction.model_version,
              application_id: prediction.application_id,
            };
          } catch (error) {
            return {
              success: false,
              error:
                error instanceof Error
                  ? error.message
                  : "Failed to analyze application",
            };
          }
        },
      },
    },
    stopWhen: stepCountIs(5),
  });

  return result.toUIMessageStreamResponse({
    sendSources: true,
    sendReasoning: true,
  });
}
