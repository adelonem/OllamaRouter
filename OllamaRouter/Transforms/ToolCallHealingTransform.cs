using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;

namespace OllamaRouter.Transforms;

/// <summary>
/// Protocol-level sanitizer for tool calling responses from local/open-weights models.
/// Strips markdown fences inside function arguments, promotes pseudo &lt;tool_call&gt; tags from content,
/// injects missing tool call IDs, and repairs trailing commas or unescaped control characters.
/// Uses source-generated regex for Native AOT compatibility.
/// </summary>
public static partial class ToolCallHealingTransform
{
    [GeneratedRegex(@"<tool_call>\s*(?<payload>\{[\s\S]*?\})\s*</tool_call>")]
    private static partial Regex ToolCallRegex();

    [GeneratedRegex(@"Action:\s*(?<name>[a-zA-Z0-9_\-\.]+)\s*\nAction Input:\s*(?<payload>\{[\s\S]*?\})")]
    private static partial Regex ActionToolCallRegex();

    [GeneratedRegex(@",\s*([\}\]])")]
    private static partial Regex TrailingCommasRegex();

    /// <summary>
    /// Transforms the given JSON node representing a chat completion response or chunk.
    /// Supports both OpenAI format (choices[].message / choices[].delta) and Ollama native format (message).
    /// </summary>
    public static JsonNode HealChatCompletionResponse(JsonNode rootNode, ILogger? logger = null)
    {
        // Support OpenAI format: choices[].message or choices[].delta
        if (rootNode["choices"] is JsonArray choices)
        {
            foreach (var choice in choices)
            {
                if (choice?["message"] is JsonObject message)
                {
                    HealMessage(message, logger);
                }
                else if (choice?["delta"] is JsonObject delta)
                {
                    HealMessage(delta, logger);
                }
            }
        }
        // Support Ollama native format: message
        else if (rootNode["message"] is JsonObject ollamaMessage)
        {
            HealMessage(ollamaMessage, logger);
        }

        return rootNode;
    }

    /// <summary>
    /// Parses a JSON response string, heals tool calling payloads, and returns the healed JSON string.
    /// Falls back to the original string if parsing fails.
    /// </summary>
    public static string HealResponseText(string json, ILogger? logger = null)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return json;
        }

        try
        {
            var node = JsonNode.Parse(json);
            if (node != null)
            {
                HealChatCompletionResponse(node, logger);
                return node.ToJsonString();
            }
        }
        catch
        {
            // Retain original string on parse error.
        }

        return json;
    }

    private static void HealMessage(JsonObject message, ILogger? logger)
    {
        // 1. Check for pseudo-tool calls in content if tool_calls is empty
        var toolCalls = message["tool_calls"] as JsonArray;

        string? content = null;
        if (message.TryGetPropertyValue("content", out var contentNode) &&
            contentNode is JsonValue cVal &&
            cVal.TryGetValue<string>(out var strVal))
        {
            content = strVal;
        }

        if ((toolCalls == null || toolCalls.Count == 0) && !string.IsNullOrWhiteSpace(content))
        {
            var matches = ToolCallRegex().Matches(content);
            if (matches.Count > 0)
            {
                var newToolCalls = new JsonArray();
                var updatedContent = content;

                foreach (Match match in matches)
                {
                    try
                    {
                        var payloadNode = JsonNode.Parse(match.Groups["payload"].Value);
                        var toolName = payloadNode?["name"]?.GetValue<string>();
                        var toolArgs = payloadNode?["arguments"] is JsonValue argVal && argVal.TryGetValue<string>(out var strArg)
                            ? strArg
                            : payloadNode?["arguments"]?.ToJsonString() ?? "{}";

                        if (!string.IsNullOrEmpty(toolName))
                        {
                            var generatedId = $"call_{Guid.NewGuid():N}"[..12];
                            newToolCalls.Add((JsonNode)new JsonObject
                            {
                                ["id"] = generatedId,
                                ["type"] = "function",
                                ["function"] = new JsonObject
                                {
                                    ["name"] = toolName,
                                    ["arguments"] = SanitizeArgumentsJson(toolArgs, logger, toolName)
                                }
                            });
                            updatedContent = updatedContent.Replace(match.Value, "");
                            logger?.LogInformation("Promoted pseudo-tool-call from content into structured tool call for '{ToolName}' (id: {ToolCallId})", toolName, generatedId);
                        }
                    }
                    catch (Exception ex)
                    {
                        logger?.LogWarning(ex, "Failed to parse pseudo-tool-call payload in content: {Payload}", match.Groups["payload"].Value);
                    }
                }

                if (newToolCalls.Count > 0)
                {
                    toolCalls = newToolCalls;
                    message["tool_calls"] = toolCalls;
                    message["content"] = updatedContent.Trim();
                }
            }
            else
            {
                var actionMatch = ActionToolCallRegex().Match(content);
                if (actionMatch.Success)
                {
                    try
                    {
                        var toolName = actionMatch.Groups["name"].Value.Trim();
                        var toolArgs = actionMatch.Groups["payload"].Value.Trim();
                        if (!string.IsNullOrEmpty(toolName))
                        {
                            var generatedId = $"call_{Guid.NewGuid():N}"[..12];
                            toolCalls = new JsonArray
                            {
                                (JsonNode)new JsonObject
                                {
                                    ["id"] = generatedId,
                                    ["type"] = "function",
                                    ["function"] = new JsonObject
                                    {
                                        ["name"] = toolName,
                                        ["arguments"] = SanitizeArgumentsJson(toolArgs, logger, toolName)
                                    }
                                }
                            };
                            message["tool_calls"] = toolCalls;
                            message["content"] = content.Replace(actionMatch.Value, "").Trim();
                            logger?.LogInformation("Promoted Action pseudo-tool-call from content into structured tool call for '{ToolName}' (id: {ToolCallId})", toolName, generatedId);
                        }
                    }
                    catch (Exception ex)
                    {
                        logger?.LogWarning(ex, "Failed to parse Action tool call payload in content: {Payload}", actionMatch.Groups["payload"].Value);
                    }
                }
            }
        }

        // 2. Sanitize existing tool_calls
        if (toolCalls != null)
        {
            foreach (var tc in toolCalls)
            {
                if (tc is JsonObject tcObj)
                {
                    var toolName = tcObj["function"]?["name"]?.GetValue<string>() ?? "unknown";

                    // Missing ID fix
                    if (string.IsNullOrWhiteSpace(tcObj["id"]?.GetValue<string>()))
                    {
                        var generatedId = $"call_{Guid.NewGuid():N}"[..12];
                        tcObj["id"] = generatedId;
                        logger?.LogInformation("Injected missing tool call ID '{ToolCallId}' for tool '{ToolName}'", generatedId, toolName);
                    }

                    if (string.IsNullOrWhiteSpace(tcObj["type"]?.GetValue<string>()))
                    {
                        tcObj["type"] = "function";
                    }

                    // Arguments fence stripping & repair
                    if (tcObj["function"] is JsonObject funcObj && funcObj["arguments"] != null)
                    {
                        if (funcObj["arguments"] is JsonValue jVal && jVal.TryGetValue<string>(out var strArgVal))
                        {
                            var cleanedArgs = SanitizeArgumentsJson(strArgVal, logger, toolName);
                            funcObj["arguments"] = cleanedArgs;
                        }
                        else
                        {
                            var rawArgs = funcObj["arguments"]!.ToJsonString();
                            var cleanedArgs = SanitizeArgumentsJson(rawArgs, logger, toolName);
                            if (cleanedArgs != rawArgs)
                            {
                                try
                                {
                                    funcObj["arguments"] = JsonNode.Parse(cleanedArgs);
                                }
                                catch
                                {
                                    funcObj["arguments"] = cleanedArgs;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Sanitizes the raw function arguments JSON string by stripping leading and trailing
    /// markdown fences (```json ... ``` or ``` ... ```), trimming whitespace, removing trailing
    /// commas, and repairing unescaped control characters in string values.
    /// </summary>
    public static string SanitizeArgumentsJson(string raw, ILogger? logger = null, string? toolName = null)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return "{}";
        }

        var s = raw.Trim();

        // Strip ```json ... ``` or ``` ... ```
        if (s.StartsWith("```"))
        {
            var firstNewline = s.IndexOf('\n');
            if (firstNewline >= 0)
            {
                s = s.Substring(firstNewline + 1);
            }
            else if (s.StartsWith("```json", StringComparison.OrdinalIgnoreCase))
            {
                s = s.Substring(7);
            }
            else
            {
                s = s.Substring(3);
            }

            if (s.EndsWith("```"))
            {
                s = s.Substring(0, s.Length - 3);
            }

            s = s.Trim();
            logger?.LogInformation("Stripped markdown code fences from arguments of tool '{ToolName}'", toolName ?? "unknown");
        }

        // Remove trailing commas before } or ]
        var afterCommas = TrailingCommasRegex().Replace(s, "$1");
        if (afterCommas != s)
        {
            logger?.LogDebug("Repaired trailing commas in arguments of tool '{ToolName}'", toolName ?? "unknown");
            s = afterCommas;
        }

        // Validate JSON and attempt to escape unescaped control chars if parsing fails
        try
        {
            using var _ = JsonDocument.Parse(s);
        }
        catch (JsonException)
        {
            var escaped = EscapeControlCharsInStrings(s);
            if (escaped != s)
            {
                logger?.LogInformation("Repaired unescaped control characters in arguments of tool '{ToolName}'", toolName ?? "unknown");
                s = escaped;
            }
        }

        return s;
    }

    private static string EscapeControlCharsInStrings(string s)
    {
        var sb = new StringBuilder(s.Length + 16);
        bool inString = false;
        bool escaped = false;

        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (escaped)
            {
                sb.Append(c);
                escaped = false;
                continue;
            }

            if (c == '\\')
            {
                escaped = true;
                sb.Append(c);
                continue;
            }

            if (c == '"')
            {
                inString = !inString;
                sb.Append(c);
                continue;
            }

            if (inString)
            {
                switch (c)
                {
                    case '\n':
                        sb.Append("\\n");
                        continue;
                    case '\r':
                        sb.Append("\\r");
                        continue;
                    case '\t':
                        sb.Append("\\t");
                        continue;
                    case < ' ':
                        sb.Append($"\\u{(int)c:x4}");
                        continue;
                }
            }

            sb.Append(c);
        }

        return sb.ToString();
    }
}
