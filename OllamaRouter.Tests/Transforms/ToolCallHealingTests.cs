using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;
using Moq;
using OllamaRouter.Transforms;

namespace OllamaRouter.Tests.Transforms;

public class ToolCallHealingTests
{
    [Fact]
    public void HealChatCompletionResponse_OpenAiFormat_StripsMarkdownFencesFromArguments()
    {
        var inputJson = """
        {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "edit_file",
                                    "arguments": "```json\n{\n  \"path\": \"src/main.cs\"\n}\n```"
                                }
                            }
                        ]
                    }
                }
            ]
        }
        """;

        var root = JsonNode.Parse(inputJson)!;
        ToolCallHealingTransform.HealChatCompletionResponse(root);

        var args = root["choices"]?[0]?["message"]?["tool_calls"]?[0]?["function"]?["arguments"]?.GetValue<string>();
        Assert.NotNull(args);
        Assert.DoesNotContain("```", args);
        Assert.Equal("{\n  \"path\": \"src/main.cs\"\n}", args);
    }

    [Fact]
    public void HealChatCompletionResponse_OllamaFormat_StripsMarkdownFencesFromArguments()
    {
        var inputJson = """
        {
            "model": "qwen2.5-coder:32b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": "```json\n{\"path\":\"a.cs\"}\n```"
                        }
                    }
                ]
            }
        }
        """;

        var root = JsonNode.Parse(inputJson)!;
        ToolCallHealingTransform.HealChatCompletionResponse(root);

        var args = root["message"]?["tool_calls"]?[0]?["function"]?["arguments"]?.GetValue<string>();
        Assert.NotNull(args);
        Assert.Equal("{\"path\":\"a.cs\"}", args);
    }

    [Fact]
    public void HealChatCompletionResponse_PseudoToolCallInContent_PromotesToStructuredToolCalls()
    {
        var inputJson = """
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I will inspect the file now.\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.cs\"}}</tool_call>\nPlease wait.",
                        "tool_calls": []
                    }
                }
            ]
        }
        """;

        var root = JsonNode.Parse(inputJson)!;
        ToolCallHealingTransform.HealChatCompletionResponse(root);

        var message = root["choices"]?[0]?["message"] as JsonObject;
        Assert.NotNull(message);

        var toolCalls = message["tool_calls"] as JsonArray;
        Assert.NotNull(toolCalls);
        Assert.Single(toolCalls);

        var call = toolCalls[0] as JsonObject;
        Assert.NotNull(call);
        Assert.StartsWith("call_", call["id"]?.GetValue<string>());
        Assert.Equal("function", call["type"]?.GetValue<string>());

        var func = call["function"] as JsonObject;
        Assert.NotNull(func);
        Assert.Equal("read_file", func["name"]?.GetValue<string>());
        Assert.Equal("{\"path\":\"src/main.cs\"}", func["arguments"]?.GetValue<string>());

        // Content must be stripped of the <tool_call> tag
        var remainingContent = message["content"]?.GetValue<string>();
        Assert.NotNull(remainingContent);
        Assert.DoesNotContain("<tool_call>", remainingContent);
        Assert.DoesNotContain("</tool_call>", remainingContent);
        Assert.Contains("I will inspect the file now.", remainingContent);
        Assert.Contains("Please wait.", remainingContent);
    }

    [Fact]
    public void HealChatCompletionResponse_MissingToolCallId_GeneratesValidId()
    {
        var inputJson = """
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": "{\"command\":\"dir\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        }
        """;

        var root = JsonNode.Parse(inputJson)!;
        ToolCallHealingTransform.HealChatCompletionResponse(root);

        var id = root["choices"]?[0]?["message"]?["tool_calls"]?[0]?["id"]?.GetValue<string>();
        Assert.False(string.IsNullOrWhiteSpace(id));
        Assert.StartsWith("call_", id);
        Assert.Equal(12, id.Length);
    }

    [Fact]
    public void SanitizeArgumentsJson_TrailingCommas_RepairsSyntax()
    {
        var inputWithTrailingCommas = """
        {
            "items": ["one", "two",],
            "config": {
                "flag": true,
            },
        }
        """;

        var sanitized = ToolCallHealingTransform.SanitizeArgumentsJson(inputWithTrailingCommas);
        Assert.DoesNotContain(",]", sanitized);
        Assert.DoesNotContain(",}", sanitized);

        // Must now be parseable as valid JSON
        var parsed = JsonNode.Parse(sanitized);
        Assert.NotNull(parsed);
    }

    [Fact]
    public void SanitizeArgumentsJson_EmptyOrWhitespace_ReturnsEmptyObject()
    {
        Assert.Equal("{}", ToolCallHealingTransform.SanitizeArgumentsJson(""));
        Assert.Equal("{}", ToolCallHealingTransform.SanitizeArgumentsJson("   \n\t"));
    }

    [Fact]
    public void SanitizeArgumentsJson_FencesWithoutNewline_StripsFencesCleanly()
    {
        var sanitized = ToolCallHealingTransform.SanitizeArgumentsJson("```json{\"path\":\"a.cs\"}```");
        Assert.Equal("{\"path\":\"a.cs\"}", sanitized);
    }

    [Fact]
    public void HealChatCompletionResponse_LogsRepairOperations()
    {
        var inputJson = """
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<tool_call>{\"name\":\"read\",\"arguments\":\"```json\\n{\\\"path\\\":\\\"a.cs\\\",}\\n```\"}</tool_call>"
                    }
                }
            ]
        }
        """;

        var loggerMock = new Mock<ILogger>();
        var root = JsonNode.Parse(inputJson)!;
        ToolCallHealingTransform.HealChatCompletionResponse(root, loggerMock.Object);

        // Verify logger was invoked for repair operations
        loggerMock.Verify(
            l => l.Log(
                LogLevel.Information,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Promoted pseudo-tool-call") || v.ToString()!.Contains("Stripped markdown")),
                It.IsAny<Exception?>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce);
    }
}
