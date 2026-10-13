using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Moq;
using OllamaRouter.Endpoints;
using OllamaRouter.Options;
using OllamaRouter.Services;
using MsOptions = Microsoft.Extensions.Options.Options;

namespace OllamaRouter.Tests.Endpoints;

public class ModelCatalogTests
{
    private static async Task<(WebApplication App, HttpClient Client, Mock<IOllamaModelCatalogClient> CatalogClient)> CreateTestAppAsync(
        Dictionary<string, ModelThresholds>? models = null)
    {
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");

        var catalogClientMock = new Mock<IOllamaModelCatalogClient>();
        builder.Services.AddSingleton(catalogClientMock.Object);

        var routerOptions = new OllamaRouterOptions
        {
            LocalUrl = "http://127.0.0.1:11435",
            RemoteUrl = "http://192.168.1.25:11434",
            Models = models ?? new()
        };
        builder.Services.AddSingleton(MsOptions.Create(routerOptions));

        var app = builder.Build();
        app.MapOllamaRouterEndpoints();
        await app.StartAsync();

        var client = new HttpClient { BaseAddress = new Uri(app.Urls.First()) };
        return (app, client, catalogClientMock);
    }

    [Fact]
    public async Task GetApiTags_WithConfiguredCloudModel_AppendsCloudModelToTags()
    {
        var models = new Dictionary<string, ModelThresholds>
        {
            ["qwen2.5-coder:32b"] = new()
            {
                CloudModel = "deepseek-v3.1:671b-cloud"
            }
        };

        var (app, client, catalogClient) = await CreateTestAppAsync(models);
        try
        {
            catalogClient.Setup(c => c.GetMergedTagsAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(new JsonArray(
                    new JsonObject { ["name"] = "qwen2.5-coder:32b", ["model"] = "qwen2.5-coder:32b" }
                ));

            var response = await client.GetAsync("/api/tags");
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);

            var json = await response.Content.ReadFromJsonAsync<JsonObject>();
            Assert.NotNull(json);

            var modelsArray = json["models"]?.AsArray();
            Assert.NotNull(modelsArray);
            Assert.Equal(2, modelsArray.Count);

            var cloudEntry = modelsArray.FirstOrDefault(m => m?["name"]?.GetValue<string>() == "deepseek-v3.1:671b-cloud");
            Assert.NotNull(cloudEntry);
            Assert.Equal(0, cloudEntry["size"]?.GetValue<long>());
            Assert.Equal("cloud", cloudEntry["details"]?["family"]?.GetValue<string>());
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public async Task GetV1Models_WithConfiguredCloudModel_AppendsCloudModelToData()
    {
        var models = new Dictionary<string, ModelThresholds>
        {
            ["llama3.2:latest"] = new()
            {
                CloudModel = "glm-5.3-flash:cloud"
            }
        };

        var (app, client, catalogClient) = await CreateTestAppAsync(models);
        try
        {
            catalogClient.Setup(c => c.GetMergedOpenAIModelsAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(new JsonArray(
                    new JsonObject { ["id"] = "llama3.2:latest", ["object"] = "model" }
                ));

            var response = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);

            var json = await response.Content.ReadFromJsonAsync<JsonObject>();
            Assert.NotNull(json);

            var dataArray = json["data"]?.AsArray();
            Assert.NotNull(dataArray);
            Assert.Equal(2, dataArray.Count);

            var cloudEntry = dataArray.FirstOrDefault(m => m?["id"]?.GetValue<string>() == "glm-5.3-flash:cloud");
            Assert.NotNull(cloudEntry);
            Assert.Equal("cloud", cloudEntry["owned_by"]?.GetValue<string>());
            Assert.Equal("model", cloudEntry["object"]?.GetValue<string>());
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }
}
