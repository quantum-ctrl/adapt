export class DataLoader {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
    }

    async listFiles() {
        const response = await fetch(`${this.baseUrl}/files`);
        if (!response.ok) throw new Error("Failed to fetch file list");
        const data = await response.json();
        return data.files;
    }

    async loadMetadata(path) {
        const response = await fetch(`${this.baseUrl}/load?path=${encodeURIComponent(path)}`);
        if (!response.ok) {
            // Try to get detailed error message from server
            let errorMsg = "Failed to load metadata";
            try {
                const errorData = await response.json();
                if (errorData.detail) {
                    errorMsg = errorData.detail;
                }
            } catch (e) {
                // If parsing fails, use status text
                errorMsg = `Failed to load metadata: ${response.status} ${response.statusText}`;
            }
            throw new Error(errorMsg);
        }
        return await response.json();
    }

    async loadData(path) {
        const response = await fetch(`${this.baseUrl}/data?path=${encodeURIComponent(path)}`);
        if (!response.ok) throw new Error("Failed to load data");

        const buffer = await response.arrayBuffer();
        // Assuming float32 data as per server implementation
        return new Float32Array(buffer);
    }

    /**
     * Load session data from ~/.adapt/session.json via the /api/session endpoint.
     * 
     * This is called when the URL contains ?session=1, indicating that ADAPT Browser
     * has written a session file that should be auto-loaded.
     * 
     * @returns {Object|null} Session data containing file_path and metadata, or null if no session
     */
    async loadSession() {
        try {
            const response = await fetch(`${this.baseUrl}/session`);
            if (!response.ok) {
                if (response.status === 404) {
                    console.log("No active session found");
                    return null;
                }
                throw new Error(`Failed to load session: ${response.status}`);
            }
            const data = await response.json();
            console.log("Session loaded:", data);
            return data;
        } catch (error) {
            console.warn("Session load error:", error.message);
            return null;
        }
    }
}

