import { promises as fs } from "fs";
import path from "path";

interface Turn {
  turn_number: number;
  role: string;
  content: string;
  timestamp: string;
}

interface Conversation {
  conversation_id: string;
  scenario_id: string;
  generator_model: string;
  patient_simulator: string;
  total_turns: number;
  generation_params: {
    physician_temperature: number;
    patient_temperature: number;
    max_tokens_per_turn: number;
    num_turns: number;
  };
  created_at: string;
  turns: Turn[];
}

async function getAvailableFiles(): Promise<string[]> {
  const dataDir = path.join(process.cwd(), "..");
  const files: string[] = [];

  try {
    const entries = await fs.readdir(dataDir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const subDir = path.join(dataDir, entry.name);
        const subEntries = await fs.readdir(subDir);
        for (const file of subEntries) {
          if (file.endsWith(".json") && file.includes("conversations")) {
            files.push(`${entry.name}/${file}`);
          }
        }
      }
    }
  } catch {
    // Ignore errors
  }

  return files;
}

async function getConversations(
  filePath: string
): Promise<Conversation[] | null> {
  try {
    // Resolve path relative to parent directory (where the data files are)
    const fullPath = path.join(process.cwd(), "..", filePath);

    // Security check: ensure path is within allowed directory
    const resolvedPath = path.resolve(fullPath);
    const parentDir = path.resolve(process.cwd(), "..");
    if (!resolvedPath.startsWith(parentDir)) {
      return null;
    }

    const fileContent = await fs.readFile(resolvedPath, "utf-8");
    const data = JSON.parse(fileContent);
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString);
    return date.toLocaleString();
  } catch {
    return dateString;
  }
}

export default async function Home({
  searchParams,
}: {
  searchParams: { file?: string };
}) {
  const file = searchParams.file;

  if (!file) {
    const availableFiles = await getAvailableFiles();

    return (
      <div className="container">
        <h1>Conversation Visualizer</h1>
        <div className="no-file">
          <h2>No file specified</h2>
          <p>
            Add a <code>?file=</code> query parameter to view conversations.
          </p>
          <p>
            Example: <code>?file=feb_1_convos_gpt/gpt-4_conversations.json</code>
          </p>

          {availableFiles.length > 0 && (
            <div className="file-list">
              <h3>Available files:</h3>
              <ul>
                {availableFiles.map((f) => (
                  <li key={f}>
                    <a href={`?file=${encodeURIComponent(f)}`}>{f}</a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  }

  const conversations = await getConversations(file);

  if (!conversations) {
    return (
      <div className="container">
        <h1>Conversation Visualizer</h1>
        <div className="error">
          <p>Could not load file: {file}</p>
          <p>Make sure the file exists and contains valid JSON.</p>
        </div>
      </div>
    );
  }

  const totalTurns = conversations.reduce((sum, c) => sum + c.total_turns, 0);

  return (
    <div className="container">
      <h1>Conversation Visualizer</h1>

      <div className="file-info">
        <strong>File:</strong> {file}
      </div>

      <div className="stats">
        <div className="stat">
          <div className="stat-value">{conversations.length}</div>
          <div className="stat-label">Conversations</div>
        </div>
        <div className="stat">
          <div className="stat-value">{totalTurns}</div>
          <div className="stat-label">Total Turns</div>
        </div>
      </div>

      <div className="conversation-list">
        {conversations.map((conversation) => (
          <details key={conversation.conversation_id} className="conversation">
            <summary>
              <span>{conversation.conversation_id}</span>
              <div className="conversation-meta">
                <span>
                  <span className="label">Scenario:</span>
                  {conversation.scenario_id}
                </span>
                <span>
                  <span className="label">Turns:</span>
                  {conversation.total_turns}
                </span>
                <span>
                  <span className="label">Created:</span>
                  {formatDate(conversation.created_at)}
                </span>
              </div>
            </summary>
            <div className="turns">
              {conversation.turns.map((turn) => (
                <div
                  key={`${conversation.conversation_id}-${turn.turn_number}`}
                  className={`turn ${turn.role}`}
                >
                  <div className="turn-header">
                    <span className="turn-role">
                      {turn.role} (Turn {turn.turn_number})
                    </span>
                    <span className="turn-timestamp">
                      {formatDate(turn.timestamp)}
                    </span>
                  </div>
                  <div className="turn-content">{turn.content}</div>
                </div>
              ))}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}
