"use client";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
  MessageActions,
  MessageAction,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { Loader } from "@/components/ai-elements/loader";
import {
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";
import { useChat } from "@ai-sdk/react";
import {
  CopyIcon,
  RefreshCcwIcon,
  CreditCardIcon,
  ActivityIcon,
  CheckCircle2Icon,
  XCircleIcon,
  LogOutIcon,
  UserIcon,
} from "lucide-react";
import { isToolUIPart, getToolName } from "ai";
import { logoutAction, getSessionAction } from "@/app/actions/auth";

const CreditScoringChat = () => {
  const [input, setInput] = useState("");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  const [userEmail, setUserEmail] = useState<string | null>(null);

  const { messages, sendMessage, status, regenerate } = useChat();

  // Check API health and get user info on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch("/backend/api/v1/health");
        setApiStatus(response.ok ? "online" : "offline");
      } catch {
        setApiStatus("offline");
      }
    };

    const getUser = async () => {
      const session = await getSessionAction();
      if (session) {
        setUserEmail(session.email);
      }
    };

    checkHealth();
    getUser();
  }, []);

  const handleSubmit = (message: PromptInputMessage) => {
    if (!message.text?.trim()) return;
    sendMessage({ text: message.text });
    setInput("");
  };

  const handleLogout = async () => {
    await logoutAction();
  };

  const renderPredictionResult = (result: any) => {
    if (!result) return null;

    if (result.success) {
      const isApproved = result.prediction === "APPROVED";
      return (
        <div
          className={`rounded-lg p-4 border ${
            isApproved
              ? "bg-green-50 border-green-200"
              : "bg-red-50 border-red-200"
          }`}
        >
          <div className="flex items-center gap-2 mb-3">
            {isApproved ? (
              <CheckCircle2Icon className="size-5 text-green-600" />
            ) : (
              <XCircleIcon className="size-5 text-red-600" />
            )}
            <span
              className={`font-semibold ${
                isApproved ? "text-green-700" : "text-red-700"
              }`}
            >
              {result.prediction}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">Confidence:</span>
              <span className="ml-2 font-medium">{result.confidence}%</span>
            </div>
            <div>
              <span className="text-gray-500">Risk Score:</span>
              <span className="ml-2 font-medium">{result.risk_score}%</span>
            </div>
            <div>
              <span className="text-gray-500">Approval Prob:</span>
              <span className="ml-2 font-medium text-green-600">
                {result.approval_probability}%
              </span>
            </div>
            <div>
              <span className="text-gray-500">Rejection Prob:</span>
              <span className="ml-2 font-medium text-red-600">
                {result.rejection_probability}%
              </span>
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-400">
            Model: {result.model_version} | ID:{" "}
            {result.application_id?.slice(0, 8)}...
          </div>
        </div>
      );
    } else {
      return (
        <div className="rounded-lg p-4 bg-yellow-50 border border-yellow-200">
          <span className="text-yellow-700">Error: {result.error}</span>
        </div>
      );
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 relative size-full h-screen">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <CreditCardIcon className="size-8 text-blue-600" />
          <div>
            <h1 className="text-xl font-bold text-gray-900">
              Credit Score Assistant
            </h1>
            <p className="text-sm text-gray-500">
              AI-powered credit application analysis
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* API Status */}
          <div className="flex items-center gap-2">
            <ActivityIcon
              className={`size-4 ${
                apiStatus === "online"
                  ? "text-green-500"
                  : apiStatus === "offline"
                    ? "text-red-500"
                    : "text-yellow-500"
              }`}
            />
            <span className="text-sm text-gray-500">
              {apiStatus === "checking"
                ? "Checking API..."
                : apiStatus === "online"
                  ? "API Online"
                  : "API Offline"}
            </span>
          </div>

          {/* User Info & Logout */}
          {userEmail && (
            <div className="flex items-center gap-2 border-l pl-4">
              <UserIcon className="size-4 text-gray-500" />
              <span className="text-sm text-gray-600">{userEmail}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleLogout}
                className="text-gray-500 hover:text-red-600"
              >
                <LogOutIcon className="size-4" />
              </Button>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col h-[calc(100%-5rem)]">
        <Conversation className="h-full">
          <ConversationContent>
            {/* Welcome message if no messages */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center p-8">
                <CreditCardIcon className="size-16 text-blue-200 mb-4" />
                <h2 className="text-lg font-semibold text-gray-700 mb-2">
                  Welcome to Credit Score Assistant
                </h2>
                <p className="text-gray-500 max-w-md mb-6">
                  I can help you analyze credit applications and understand your
                  creditworthiness. Tell me about your financial situation and
                  I&apos;ll provide a prediction.
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {[
                    "I want to apply for a personal loan",
                    "Can you help me check my eligibility?",
                    "What information do you need for a credit check?",
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => {
                        setInput(suggestion);
                        sendMessage({ text: suggestion });
                      }}
                      className="px-4 py-2 text-sm bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100 transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.map((message) => (
              <div key={message.id}>
                {message.parts.map((part, i) => {
                  // Handle tool calls
                  if (isToolUIPart(part)) {
                    const toolName = getToolName(part);
                    const result =
                      part.state === "output-available"
                        ? part.output
                        : undefined;

                    return (
                      <Tool key={`${message.id}-${i}`} defaultOpen>
                        <ToolHeader
                          title={
                            toolName === "analyze_credit_application"
                              ? "Credit Analysis"
                              : toolName
                          }
                          type={part.type}
                          state={part.state}
                        />
                        <ToolContent>
                          <ToolInput input={part.input} />
                          {part.state === "output-available" && (
                            <div className="p-4">
                              {renderPredictionResult(result)}
                            </div>
                          )}
                          {part.state === "output-error" && (
                            <ToolOutput
                              output={null}
                              errorText={part.errorText}
                            />
                          )}
                        </ToolContent>
                      </Tool>
                    );
                  }

                  // Handle text messages
                  if (part.type === "text") {
                    return (
                      <Message key={`${message.id}-${i}`} from={message.role}>
                        <MessageContent>
                          <MessageResponse>{part.text}</MessageResponse>
                        </MessageContent>
                        {message.role === "assistant" && (
                          <MessageActions>
                            <MessageAction
                              onClick={() => regenerate()}
                              label="Retry"
                            >
                              <RefreshCcwIcon className="size-3" />
                            </MessageAction>
                            <MessageAction
                              onClick={() =>
                                navigator.clipboard.writeText(part.text)
                              }
                              label="Copy"
                            >
                              <CopyIcon className="size-3" />
                            </MessageAction>
                          </MessageActions>
                        )}
                      </Message>
                    );
                  }

                  return null;
                })}
              </div>
            ))}
            {status === "submitted" && <Loader />}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        {/* Input */}
        <PromptInput onSubmit={handleSubmit} className="mt-4">
          <PromptInputBody>
            <PromptInputTextarea
              onChange={(e) => setInput(e.target.value)}
              value={input}
              placeholder="Tell me about your loan application..."
              disabled={apiStatus === "offline"}
            />
          </PromptInputBody>
          <PromptInputFooter>
            <PromptInputTools>
              <PromptInputButton variant="ghost" disabled>
                <CreditCardIcon size={16} />
                <span>Credit Analysis</span>
              </PromptInputButton>
            </PromptInputTools>
            <PromptInputSubmit
              disabled={
                (!input && status === "ready") || apiStatus === "offline"
              }
              status={status}
            />
          </PromptInputFooter>
        </PromptInput>
      </div>
    </div>
  );
};

export default CreditScoringChat;
