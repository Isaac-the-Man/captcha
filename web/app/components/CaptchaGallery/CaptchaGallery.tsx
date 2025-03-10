import { Box, Typography } from "@mui/material";
import { CaptchaDisplay } from "../CaptchaDisplay";


export interface CaptchaGalleryItemProps {
  captchaText: string;
  captchaSolvedText: string;
  model: string;
  captchaImageUrl: string;
}

export function CaptchaGalleryItem({
  captchaText,
  captchaSolvedText,
  model,
  captchaImageUrl,
}: CaptchaGalleryItemProps) {
  return (
    <Box
      sx={{
        p: 2,
        bgcolor: "white",
        boxShadow: 2,
        borderRadius: 1,
      }}
    >
      <Typography>
        Label: {captchaText}<br />
        Prediction: {captchaSolvedText}<br />
        Model: {model}
      </Typography>
      <CaptchaDisplay
        status="success"
        captchaText={captchaText}
        captchaImageUrl={captchaImageUrl}
      />
    </Box>
  );
}

export interface CaptchaGalleryProps {
  items: CaptchaGalleryItemProps[];
}

export function CaptchaGallery({
  items,
}: CaptchaGalleryProps) {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "row",
        flexWrap: "wrap",
        gap: 2,
      }}
    >
      {items.length === 0 && (
        <Typography
          variant="body2"
          color="text.secondary"
        >
          No Captcha solved yet.
        </Typography>
      )}
      {items.map((item, index) => (
        <CaptchaGalleryItem
          key={index}
          captchaText={item.captchaText}
          captchaSolvedText={item.captchaSolvedText}
          model={item.model}
          captchaImageUrl={item.captchaImageUrl}
        />
      ))}
    </Box>
  );
}