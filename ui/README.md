# Credit Score Assistant UI

A Next.js frontend for the Credit Score ML API, featuring an AI-powered chat interface built with the Vercel AI SDK.

## Features

- **AI-Powered Chat**: Natural language interface for credit applications
- **Real-time Predictions**: Streaming responses with tool execution
- **FastAPI Integration**: Connects to the ML backend for credit scoring
- **Modern UI Components**: Built with ai-elements and shadcn/ui

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│  Next.js API     │────▶│   FastAPI       │
│   (ai-elements) │     │  Route Handler   │     │   Backend       │
│                 │     │  (streamText)    │     │   (/api/v1)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌──────────────────┐
                        │   OpenAI API     │
                        │   (LLM Chat)     │
                        └──────────────────┘
```

## Getting Started

### Prerequisites

- Node.js 18+
- pnpm (recommended) or npm
- Running FastAPI backend (default: http://localhost:8000)
- OpenAI API key

### Setup

1. Copy the environment file:

```bash
cp .env.example .env.local
```

2. Configure environment variables in `.env.local`:

```env
API_BASE_URL=http://localhost:8000
OPENAI_API_KEY=your-openai-api-key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Install dependencies:

```bash
pnpm install
```

4. Run the development server:

```bash
pnpm dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## API Integration

The UI connects to the FastAPI backend through:

- **Proxy Route**: `/backend/*` proxies to FastAPI backend
- **Chat API**: `/api/chat` handles AI chat with tool execution
- **Health Check**: Monitors backend status in real-time

### Credit Analysis Tool

The chat interface includes a tool that calls the FastAPI `/api/v1/predict` endpoint:

```typescript
tools: {
  analyze_credit_application: {
    description: 'Analyze a credit application',
    inputSchema: creditApplicationSchema,
    execute: async (application) => {
      // Calls FastAPI predict endpoint
      return await makeCreditPrediction(application);
    },
  },
}
```

## Key Files

- `src/app/page.tsx` - Main chat interface
- `src/app/api/chat/route.ts` - API route with tool definitions
- `src/lib/credit-api.ts` - FastAPI client service
- `src/lib/types.ts` - TypeScript type definitions

## Development

```bash
# Run development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start

# Lint code
pnpm lint

# Format code
pnpm format
```

## Learn More

- [Vercel AI SDK](https://ai-sdk.dev/docs) - AI SDK documentation
- [Next.js Docs](https://nextjs.org/docs) - Next.js documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/) - FastAPI documentation
