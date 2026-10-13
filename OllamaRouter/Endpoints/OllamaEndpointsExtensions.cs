using System.Text.Json.Nodes;
using Microsoft.Extensions.Options;
using OllamaRouter.Options;
using OllamaRouter.Services;

namespace OllamaRouter.Endpoints;

/// <summary>
/// Hybrid endpoints that short-circuit YARP to merge local and remote model catalogs
/// before returning them to the client.
/// </summary>
public static class OllamaEndpointsExtensions
{
    public static WebApplication MapOllamaRouterEndpoints(this WebApplication app)
    {
        app.MapGet("/api/tags", async (IOllamaModelCatalogClient catalogClient, IOptions<OllamaRouterOptions> options) =>
        {
            var mergedModels = await catalogClient.GetMergedTagsAsync(options.Value.LocalUrl, options.Value.RemoteUrl);

            if (options.Value.Models != null)
            {
                foreach (var (_, modelOpts) in options.Value.Models)
                {
                    if (!string.IsNullOrWhiteSpace(modelOpts.CloudModel))
                    {
                        var cloudName = modelOpts.CloudModel;
                        if (!mergedModels.Any(m => string.Equals(m?["name"]?.GetValue<string>(), cloudName, StringComparison.OrdinalIgnoreCase) ||
                                                   string.Equals(m?["model"]?.GetValue<string>(), cloudName, StringComparison.OrdinalIgnoreCase)))
                        {
                            mergedModels.Add((JsonNode)new JsonObject
                            {
                                ["name"] = cloudName,
                                ["model"] = cloudName,
                                ["modified_at"] = DateTime.UtcNow.ToString("o"),
                                ["size"] = 0,
                                ["details"] = new JsonObject
                                {
                                    ["family"] = "cloud",
                                    ["parameter_size"] = "cloud",
                                    ["quantization_level"] = "cloud"
                                }
                            });
                        }
                    }
                }
            }

            var response = new JsonObject
            {
                ["models"] = mergedModels
            };

            return Results.Content(response.ToJsonString(), "application/json");
        });

        app.MapGet("/api/ps", async (IOllamaModelCatalogClient catalogClient, IOptions<OllamaRouterOptions> options) =>
        {
            var mergedModels = await catalogClient.GetMergedRunningModelsAsync(options.Value.LocalUrl, options.Value.RemoteUrl);

            var response = new JsonObject
            {
                ["models"] = mergedModels
            };

            return Results.Content(response.ToJsonString(), "application/json");
        });

        app.MapGet("/v1/models", async (IOllamaModelCatalogClient catalogClient, IOptions<OllamaRouterOptions> options) =>
        {
            var mergedModels = await catalogClient.GetMergedOpenAIModelsAsync(options.Value.LocalUrl, options.Value.RemoteUrl);

            if (options.Value.Models != null)
            {
                foreach (var (_, modelOpts) in options.Value.Models)
                {
                    if (!string.IsNullOrWhiteSpace(modelOpts.CloudModel))
                    {
                        var cloudName = modelOpts.CloudModel;
                        if (!mergedModels.Any(m => string.Equals(m?["id"]?.GetValue<string>(), cloudName, StringComparison.OrdinalIgnoreCase)))
                        {
                            mergedModels.Add((JsonNode)new JsonObject
                            {
                                ["id"] = cloudName,
                                ["object"] = "model",
                                ["created"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                                ["owned_by"] = "cloud"
                            });
                        }
                    }
                }
            }

            var response = new JsonObject
            {
                ["object"] = "list",
                ["data"] = mergedModels
            };

            return Results.Content(response.ToJsonString(), "application/json");
        });

        return app;
    }
}
