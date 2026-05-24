export class DataLoader {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
    }

    async getErrorMessage(response, fallback) {
        try {
            const errorData = await response.json();
            if (errorData.detail) {
                return typeof errorData.detail === 'string'
                    ? errorData.detail
                    : JSON.stringify(errorData.detail);
            }
            if (errorData.error) {
                return errorData.error;
            }
        } catch (e) {
            // Fall through to status-based fallback below.
        }
        return `${fallback}: ${response.status} ${response.statusText}`;
    }

    async listFiles() {
        const response = await fetch(`${this.baseUrl}/files`);
        if (!response.ok) {
            throw new Error(await this.getErrorMessage(response, "Failed to fetch file list"));
        }
        const data = await response.json();
        return data.files;
    }

    async loadMetadata(path) {
        const response = await fetch(`${this.baseUrl}/load?path=${encodeURIComponent(path)}`);
        if (!response.ok) {
            throw new Error(await this.getErrorMessage(response, "Failed to load metadata"));
        }
        return await response.json();
    }

    async loadData(path) {
        const response = await fetch(`${this.baseUrl}/data?path=${encodeURIComponent(path)}`);
        if (!response.ok) {
            throw new Error(await this.getErrorMessage(response, "Failed to load data"));
        }

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
                    return null;
                }
                throw new Error(await this.getErrorMessage(response, "Failed to load session"));
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.warn("Session load error:", error.message);
            return null;
        }
    }
}
