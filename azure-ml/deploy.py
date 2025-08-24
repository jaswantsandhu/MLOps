import uuid
from config import ml_client
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

def list_registered_models():
    """List all registered models with None-safe version handling"""
    try:
        models = list(ml_client.models.list())
        print("\nRegistered models:")
        if not models:
            print("  No models found")
            return []
        
        model_info = []
        for model in models:
            version_str = str(model.version) if model.version is not None else "None"
            print(f"  - {model.name}:{version_str}")
            model_info.append((model.name, model.version))
        return model_info
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def create_endpoint():
    endpoint_name = "iris-endpoint-" + str(uuid.uuid4())[:8]
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Iris classification endpoint",
        auth_mode="key",
        tags={"model_type": "sklearn.RandomForestClassifier"},
        location="uksouth"            # ← UK South
    )
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result().name

def deploy_model(endpoint_name):
    """Deploy the most-recent iris_model, handling VM-size quota and SKU support."""
    from datetime import datetime
    # 1) grab the latest iris_model
    models_list = list(ml_client.models.list(name="iris_model"))
    if not models_list:
        raise Exception("No iris_model registrations found.")
    def created_at(m):
        return getattr(m.creation_context, "created_at", datetime(1970,1,1))
    model = max(models_list, key=created_at)
    print(f"✅ Using iris_model created at {model.creation_context.created_at}")

    # 2) helper to deploy on a given SKU
    def try_deploy(vm_size):
        dep = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=model,
            instance_type=vm_size,
            instance_count=1,
        )
        print(f"🚀 Trying {vm_size}…")
        return ml_client.online_deployments.begin_create_or_update(dep).result()

    # 3) fallback sequence with correct names
    for sku in ("Standard_DS3_v2", "Standard_E4s_v3", "Standard_E2s_v3"):
        try:
            deployment = try_deploy(sku)
            chosen = sku
            break
        except Exception as e:
            if "not enough quota" in str(e).lower() or "not supported" in str(e).lower():
                print(f"⚠️ {sku} failed—falling back…")
                continue
            raise

    # 4) route 100% traffic
    ep = ml_client.online_endpoints.get(endpoint_name)
    ep.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(ep).result()

    print(f"✅ Model deployed successfully on {chosen}")
    return endpoint_name


def test_endpoint(endpoint_name):
    """Test the deployed endpoint"""
    # Sample iris data: [sepal_length, sepal_width, petal_length, petal_width]
    test_data = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [6.2, 3.4, 5.4, 2.3],  # Virginica
            [5.8, 2.7, 4.1, 1.0]   # Versicolor
        ]
    }
    
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name="blue",
        request_file=test_data
    )
    
    print(f"Prediction results: {response}")
    return response

if __name__ == "__main__":
    print("Step 1: Creating endpoint...")
    endpoint_name = create_endpoint()

    print("\nStep 2: Deploying model...")
    deploy_model(endpoint_name)

    print("\nStep 3: Testing endpoint...")
    test_endpoint(endpoint_name)

    print(f"\nDeployment complete! Endpoint name: {endpoint_name}")
