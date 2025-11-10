# DT_backend

Mini RAG Backend auf Basis von FastAPI. Lädt lokale `.txt`-Dateien, erzeugt Embeddings via Groq, speichert sie in ChromaDB und beantwortet Anfragen samt Sensordaten aus Unity.

## Voraussetzungen
- Python 3.10+
- Installierte Pakete: `fastapi`, `uvicorn`, `chromadb`, `requests`, `pydantic`
- Gesetzte Umgebungsvariable `GROQ_API_KEY`

## Dokumente vorbereiten
Lege deine Wissensdateien als `.txt` unter `app/documents/` ab. Bei jedem Serverstart werden alle Dateien neu eingelesen und in ChromaDB eingebettet.

## Server starten
```bash
pip install fastapi uvicorn chromadb requests pydantic
export GROQ_API_KEY="dein_geheimer_api_key"
uvicorn app.main:app --reload
```

## Unity-Integration
Beispiel für eine C#-POST-Anfrage aus Unity (UnityWebRequest):
```csharp
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class RagClient : MonoBehaviour
{
    [System.Serializable]
    private class QueryPayload
    {
        public string question;
        public SensorData sensor_data;
    }

    [System.Serializable]
    private class SensorData
    {
        public float temperature;
        public float humidity;
    }

    public IEnumerator AskBackend(string question, float temperature, float humidity)
    {
        var payload = new QueryPayload
        {
            question = question,
            sensor_data = new SensorData
            {
                temperature = temperature,
                humidity = humidity
            }
        };

        var json = JsonUtility.ToJson(payload);
        using var request = new UnityWebRequest("http://localhost:8000/query", UnityWebRequest.kHttpVerbPOST);
        var bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log($"Antwort: {request.downloadHandler.text}");
        }
        else
        {
            Debug.LogError($"Fehler: {request.error}");
        }
    }
}
```

Die Antwort enthält `answer`, `context` und die zurückgespielten `sensor_data`.
