using System.Text;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;

namespace OllamaRouter.Transforms;

/// <summary>
/// Intercepts response stream writes to apply <see cref="ToolCallHealingTransform"/>
/// to SSE, NDJSON, and buffered JSON completion responses.
/// </summary>
internal sealed class ToolCallHealingResponseStream : Stream
{
    private readonly Stream _inner;
    private readonly HttpContext _context;
    private readonly bool _enabled;
    private readonly ILogger? _logger;

    private enum StreamKind { Unknown, Sse, Ndjson, BufferedJson, PassThrough }
    private StreamKind _kind = StreamKind.Unknown;

    private readonly MemoryStream _bufferedJsonStream = new();
    private readonly StringBuilder _textBuffer = new();
    private readonly Decoder _decoder = Encoding.UTF8.GetDecoder();

    public ToolCallHealingResponseStream(Stream inner, HttpContext context, bool enabled, ILogger? logger = null)
    {
        _inner = inner;
        _context = context;
        _enabled = enabled;
        _logger = logger;
    }

    public async Task FinishAsync()
    {
        if (!_enabled)
        {
            return;
        }

        if (_kind == StreamKind.BufferedJson)
        {
            if (_bufferedJsonStream.Length > 0)
            {
                var bytes = _bufferedJsonStream.ToArray();
                if (_context.Response.StatusCode < 400)
                {
                    try
                    {
                        var json = Encoding.UTF8.GetString(bytes);
                        var node = JsonNode.Parse(json);
                        if (node != null)
                        {
                            ToolCallHealingTransform.HealChatCompletionResponse(node, _logger);
                            var healedJson = node.ToJsonString();
                            var healedBytes = Encoding.UTF8.GetBytes(healedJson);

                            if (_context.Response.Headers.ContainsKey("Content-Length"))
                            {
                                _context.Response.Headers.ContentLength = healedBytes.Length;
                            }

                            await _inner.WriteAsync(healedBytes);
                            await _inner.FlushAsync();
                            return;
                        }
                    }
                    catch
                    {
                        // Fall back to original bytes on parse error
                    }
                }

                await _inner.WriteAsync(bytes);
                await _inner.FlushAsync();
            }
        }
        else if (_kind == StreamKind.Ndjson)
        {
            if (_textBuffer.Length > 0)
            {
                var remaining = _textBuffer.ToString().Trim();
                if (remaining.Length > 0 && remaining.StartsWith('{'))
                {
                    try
                    {
                        var node = JsonNode.Parse(remaining);
                        if (node != null)
                        {
                            ToolCallHealingTransform.HealChatCompletionResponse(node, _logger);
                            remaining = node.ToJsonString();
                        }
                    }
                    catch
                    {
                        // Pass through on parse error
                    }
                }

                if (remaining.Length > 0)
                {
                    var bytes = Encoding.UTF8.GetBytes(remaining + "\n");
                    await _inner.WriteAsync(bytes);
                }

                await _inner.FlushAsync();
            }
        }
        else if (_kind == StreamKind.Sse)
        {
            if (_textBuffer.Length > 0)
            {
                var remaining = _textBuffer.ToString();
                await ProcessSseBlockAsync(remaining);
                await _inner.FlushAsync();
            }
        }
    }

    private void DetectStreamKind(ReadOnlySpan<byte> initialChunk)
    {
        if (!_enabled)
        {
            _kind = StreamKind.PassThrough;
            return;
        }

        var contentType = _context.Response.ContentType ?? "";
        if (contentType.Contains("text/event-stream", StringComparison.OrdinalIgnoreCase))
        {
            _kind = StreamKind.Sse;
            return;
        }

        if (contentType.Contains("application/x-ndjson", StringComparison.OrdinalIgnoreCase))
        {
            _kind = StreamKind.Ndjson;
            return;
        }

        if (contentType.Contains("application/json", StringComparison.OrdinalIgnoreCase))
        {
            _kind = StreamKind.BufferedJson;
            return;
        }

        // Infer from body content if content type was not yet set
        var sample = Encoding.UTF8.GetString(initialChunk.Length > 256 ? initialChunk[..256] : initialChunk).TrimStart();
        if (sample.StartsWith("data:", StringComparison.Ordinal))
        {
            _kind = StreamKind.Sse;
        }
        else if (sample.StartsWith('{') && (sample.Contains("\"done\":") || sample.Contains('\n')))
        {
            _kind = StreamKind.Ndjson;
        }
        else
        {
            _kind = StreamKind.BufferedJson;
        }
    }

    public override async Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
    {
        await WriteAsync(new ReadOnlyMemory<byte>(buffer, offset, count), cancellationToken);
    }

    public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
    {
        if (buffer.IsEmpty)
        {
            return;
        }

        if (_kind == StreamKind.Unknown)
        {
            DetectStreamKind(buffer.Span);
        }

        if (_kind == StreamKind.PassThrough || !_enabled)
        {
            await _inner.WriteAsync(buffer, cancellationToken);
            return;
        }

        if (_kind == StreamKind.BufferedJson)
        {
            _bufferedJsonStream.Write(buffer.Span);
            return;
        }

        if (_kind == StreamKind.Ndjson)
        {
            await ProcessNdjsonChunkAsync(buffer, cancellationToken);
            return;
        }

        if (_kind == StreamKind.Sse)
        {
            await ProcessSseChunkAsync(buffer, cancellationToken);
            return;
        }

        await _inner.WriteAsync(buffer, cancellationToken);
    }

    public override void Write(byte[] buffer, int offset, int count)
    {
        WriteAsync(buffer, offset, count).GetAwaiter().GetResult();
    }

    private async Task ProcessNdjsonChunkAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken)
    {
        var chars = new char[_decoder.GetCharCount(buffer.Span, false)];
        _decoder.GetChars(buffer.Span, chars, false);
        _textBuffer.Append(chars);

        int newlineIndex;
        while ((newlineIndex = IndexOfNewline(_textBuffer)) >= 0)
        {
            var line = _textBuffer.ToString(0, newlineIndex).TrimEnd('\r');
            _textBuffer.Remove(0, newlineIndex + 1);

            var trimmed = line.Trim();
            if (trimmed.Length > 0 && trimmed.StartsWith('{'))
            {
                try
                {
                    var node = JsonNode.Parse(trimmed);
                    if (node != null)
                    {
                        ToolCallHealingTransform.HealChatCompletionResponse(node, _logger);
                        line = node.ToJsonString();
                    }
                }
                catch
                {
                    // Fall back to original line
                }
            }

            var lineBytes = Encoding.UTF8.GetBytes(line + "\n");
            await _inner.WriteAsync(lineBytes, cancellationToken);
        }

        await _inner.FlushAsync(cancellationToken);
    }

    private async Task ProcessSseChunkAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken)
    {
        var chars = new char[_decoder.GetCharCount(buffer.Span, false)];
        _decoder.GetChars(buffer.Span, chars, false);
        _textBuffer.Append(chars);

        string bufferStr = _textBuffer.ToString();
        int eventEnd;
        while ((eventEnd = FindSseEventEnd(bufferStr)) >= 0)
        {
            var eventBlock = bufferStr.Substring(0, eventEnd);
            bufferStr = bufferStr.Substring(eventEnd);
            _textBuffer.Clear();
            _textBuffer.Append(bufferStr);

            await ProcessSseBlockAsync(eventBlock, cancellationToken);
        }

        await _inner.FlushAsync(cancellationToken);
    }

    private async Task ProcessSseBlockAsync(string block, CancellationToken cancellationToken = default)
    {
        var lines = block.Split('\n');
        var output = new StringBuilder();

        foreach (var rawLine in lines)
        {
            var line = rawLine.TrimEnd('\r');
            if (line.StartsWith("data:", StringComparison.Ordinal))
            {
                var payload = line["data:".Length..].Trim();
                if (payload == "[DONE]")
                {
                    output.AppendLine(line);
                }
                else if (payload.Length > 0 && payload.StartsWith('{'))
                {
                    try
                    {
                        var node = JsonNode.Parse(payload);
                        if (node != null)
                        {
                            ToolCallHealingTransform.HealChatCompletionResponse(node, _logger);
                            payload = node.ToJsonString();
                        }
                    }
                    catch
                    {
                        // Retain original payload
                    }
                    output.AppendLine($"data: {payload}");
                }
                else
                {
                    output.AppendLine(line);
                }
            }
            else
            {
                output.AppendLine(line);
            }
        }

        var blockBytes = Encoding.UTF8.GetBytes(output.ToString());
        await _inner.WriteAsync(blockBytes, cancellationToken);
    }

    private static int IndexOfNewline(StringBuilder sb)
    {
        for (int i = 0; i < sb.Length; i++)
        {
            if (sb[i] == '\n') return i;
        }
        return -1;
    }

    private static int FindSseEventEnd(string text)
    {
        int idx = text.IndexOf("\n\n", StringComparison.Ordinal);
        if (idx >= 0) return idx + 2;

        idx = text.IndexOf("\r\n\r\n", StringComparison.Ordinal);
        if (idx >= 0) return idx + 4;

        return -1;
    }

    public override void Flush() => _inner.Flush();
    public override Task FlushAsync(CancellationToken cancellationToken) => _inner.FlushAsync(cancellationToken);

    public override bool CanRead => false;
    public override bool CanSeek => false;
    public override bool CanWrite => true;
    public override long Length => throw new NotSupportedException();
    public override long Position
    {
        get => throw new NotSupportedException();
        set => throw new NotSupportedException();
    }
    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
}
