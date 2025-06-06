import React from "react";
import { Button } from "@/components/ui/button";
import { AgentSettings } from "./types";
import { RoomLayout } from "../layout/RoomLayout";
import { WaveAvatar } from "./WaveAvatar";

interface MeetingContainerProps {
  onConnect: () => void;
  agentSettings: AgentSettings;
  isConnecting: boolean;
  onSettingsChange?: (settings: AgentSettings) => void;
}

export const MeetingContainer: React.FC<MeetingContainerProps> = ({
  onConnect,
  agentSettings,
  isConnecting,
  onSettingsChange,
}) => {
  return (
    <RoomLayout
      agentSettings={agentSettings}
      onSettingsChange={onSettingsChange}
    >
      <div className="flex flex-col items-center justify-between min-h-[400px] w-[300px] h-[50%]">
        {/* Agent Avatar with Wave Animation */}
        <WaveAvatar isConnected={false} className="mb-8" />

        {/* Control Panel */}
        <div className="flex items-center space-x-6">
          {/* Connect Button */}
          <Button
            onClick={onConnect}
            disabled={isConnecting}
            className="px-8 py-3 bg-[#0b3820] hover:bg-[#0b3820] text-[#3fa16d]"
          >
            {isConnecting ? "Connecting..." : "Connect"}
          </Button>
        </div>
      </div>
    </RoomLayout>
  );
};
